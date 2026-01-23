import numpy as np
import torch
import math
import copy
import os
from torch.nn.functional import normalize
import gc
import logging
from tqdm import tqdm


TRAFFIC_STATS = {
    "FlexLoRA": {"transmit_MB": 0.0, "compute_MB": 0.0},
    "FedHera": {"transmit_MB": 0.0, "compute_MB": 0.0},
    "FedHeLLo": {"transmit_MB": 0.0, "compute_MB": 0.0},
    "FLoRA": {"transmit_MB": 0.0, "compute_MB": 0.0},
}
FEDHERA_CLIENT_STATS = {}


def reset_traffic_stats():
    for method in TRAFFIC_STATS:
        TRAFFIC_STATS[method]["transmit_MB"] = 0.0
        TRAFFIC_STATS[method]["compute_MB"] = 0.0


def get_traffic_stats():
    return {
        name: stats.copy()
        for name, stats in TRAFFIC_STATS.items()
    }

def FLoRA(selected_clients_set, output_dir, local_dataset_len_dict, epoch, client_budgets=None, layer_specs=None):
    """
    FLoRA Aggregation: Stack heterogeneous LoRA modules.
    Ref: FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations
    
    Implementation:
    1. Collect A and B matrices from all clients.
    2. Apply weighting p_k to Matrix A.
    3. Stack A (vertically) and B (horizontally).
    4. Compute B_stack @ A_stack to get the global dense update.
    5. Return the dense update so 'distribute_weight_fast' can redistribute it via SVD.
    """
    weights_array = torch.tensor(
        [local_dataset_len_dict[client_id] for client_id in selected_clients_set], dtype=torch.float32
    )
    weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)
    
    stacking_buffer = {}
    
    round_transmit_bytes = 0.0
    round_compute_bytes = 0.0
    compute_ms_per_param = 1.7e-04
    MB = 1024.0 * 1024.0

    with torch.no_grad():
        for k, client_id in tqdm(enumerate(selected_clients_set)):
            single_output_dir = os.path.join(
                output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin"
            )
            single_weights = torch.load(single_output_dir, map_location='cpu')
            
            comm_bytes = 0.0
            comp_time_ms = 0.0
            
            for key in list(single_weights.keys()):
                if 'local' in key and 'bias' not in key and 'lora_A' in key:
                    B_key = key.replace('lora_A', 'lora_B')
                    
                    base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                    
                    if base_key not in stacking_buffer:
                        stacking_buffer[base_key] = {'A': [], 'B': []}
                    
                    A_w = single_weights[key]
                    B_w = single_weights[B_key]
                    rank = A_w.shape[0] # LoRA A is [r, d_in]
                    
                    merge_rate = 16 / max(rank, 1)
                    
                    A_weighted = A_w * weights_array[k] * merge_rate
                    
                    stacking_buffer[base_key]['A'].append(A_weighted)
                    stacking_buffer[base_key]['B'].append(B_w)

                    d_in = A_w.shape[1]
                    d_out = B_w.shape[0]
                    elem_bytes = A_w.element_size()
                    bytes_this = (d_in + d_out) * rank * elem_bytes
                    round_transmit_bytes += bytes_this
                    round_compute_bytes += bytes_this
                    
                    if client_budgets is not None:
                        if layer_specs is not None and base_key in layer_specs:
                            spec = layer_specs[base_key]
                            d_out_spec = int(spec.get("d_out", d_out))
                            d_in_spec = int(spec.get("d_in", d_in))
                        else:
                            d_out_spec, d_in_spec = d_out, d_in
                        
                        comm_bytes += float(rank * (d_in_spec + d_out_spec) * 2.0)
                        comp_time_ms += float(rank * (d_in_spec + d_out_spec) * compute_ms_per_param)

            if client_budgets is not None:
                budget = client_budgets.get(int(client_id)) if isinstance(client_budgets, dict) else None
                if budget is not None:
                    B_down_bytes = float(budget.get("B_down_MB", 0.0)) * MB
                    T_ms = float(budget.get("step_ms", 0.0))
                    comm_util = (comm_bytes / B_down_bytes) if B_down_bytes > 0 else 0.0
                    comp_util = (comp_time_ms / T_ms) if T_ms > 0 else 0.0
                    logging.info(
                        "[FLoRA][epoch %d][client %s] Comm Util: %.1f%%, Comp Util: %.1f%%",
                        epoch,
                        str(client_id),
                        comm_util * 100.0,
                        comp_util * 100.0,
                    )

            del single_weights
            torch.cuda.empty_cache()

    weighted_single_weights = {}
    for base_key, matrices in stacking_buffer.items():
        if not matrices['A']: 
            continue
            
        A_stack = torch.cat(matrices['A'], dim=0)
        
        B_stack = torch.cat(matrices['B'], dim=1)
        
        merged_weight = B_stack @ A_stack
        
        weighted_single_weights[base_key] = merged_weight
        
        del A_stack, B_stack, merged_weight
        torch.cuda.empty_cache()

    TRAFFIC_STATS["FLoRA"]["transmit_MB"] += round_transmit_bytes / MB
    TRAFFIC_STATS["FLoRA"]["compute_MB"] += round_compute_bytes / MB
    logging.info(
        "[FLoRA][epoch %d] transmit_MB=%.3f compute_MB=%.3f",
        epoch,
        round_transmit_bytes / MB,
        round_compute_bytes / MB,
    )
    
    return weighted_single_weights

def FedAvg(selected_clients_set, output_dir, local_dataset_len_dict, epoch, client_budgets=None, layer_specs=None):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)
    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(client_id), "local_output_epoch_{}".format(epoch),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir, map_location='cpu')
        with torch.no_grad():
            if k == 0:
                weighted_single_weights = {key: 0 for key in
                                           single_weights.keys()}

            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}
        if client_budgets is not None:
            budget = client_budgets.get(int(client_id)) if isinstance(client_budgets, dict) else None
            if budget is not None:
                comm_bytes = 0.0
                comp_time_ms = 0.0
                for key in list(single_weights.keys()):
                    if 'local' not in key or 'bias' in key or 'lora_A' not in key:
                        continue
                    B_key = key.replace('lora_A', 'lora_B')
                    if B_key not in single_weights:
                        continue
                    A_w = single_weights[key]
                    B_w = single_weights[B_key]
                    rank = int(B_w.shape[1])
                    base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                    d_out = int(B_w.shape[0])
                    d_in = int(A_w.shape[1]) if A_w.ndim >= 2 else int(A_w.shape[-1])
                    if layer_specs is not None and base_key in layer_specs:
                        spec = layer_specs[base_key]
                        d_out = int(spec.get("d_out", d_out))
                        d_in = int(spec.get("d_in", d_in))
                    comm_bytes += float(rank * (d_in + d_out) * 2.0)
                    comp_time_ms += float(rank * (d_in + d_out) * 1.7e-04)
                B_down_bytes = float(budget.get("B_down_MB", 0.0)) * 1024.0 * 1024.0
                T_ms = float(budget.get("step_ms", 0.0))
                comm_util = (comm_bytes / B_down_bytes) if B_down_bytes > 0 else 0.0
                comp_util = (comp_time_ms / T_ms) if T_ms > 0 else 0.0
                logging.info(
                    "[FedAvg][epoch %d][client %s] Comm Util: %.1f%%, Comp Util: %.1f%%",
                    epoch,
                    str(client_id),
                    comm_util * 100.0,
                    comp_util * 100.0,
                )
        del single_weights
        gc.collect()
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return weighted_single_weights

def FedHeLLo(selected_clients_set, output_dir, local_dataset_len_dict, epoch,
             active_layers_map=None, prev_global_params=None, layer_specs=None):
    import os
    from torch.nn.functional import normalize

    del active_layers_map
    total_layers = len(layer_specs or {})

    lens = [local_dataset_len_dict[client_id] for client_id in selected_clients_set]
    weights_array = normalize(
        torch.tensor(lens, dtype=torch.float32),
        p=1, dim=0
    )

    accum = {}       
    weight_sums = {}   
    round_transmit_bytes = 0.0
    MB = 1024.0 * 1024.0

    with torch.no_grad():
        for idx, client_id in enumerate(selected_clients_set):
            single_output_dir = os.path.join(
                output_dir,
                str(client_id),
                f"local_output_epoch_{epoch}",
                "pytorch_model.bin",
            )
            if not os.path.exists(single_output_dir):
                continue

            state = torch.load(single_output_dir, map_location="cpu")
            w = float(weights_array[idx])

            for key, tensor in state.items():
                if key not in accum:
                    accum[key] = torch.zeros_like(tensor)
                    weight_sums[key] = 0.0
                accum[key] += tensor * w
                weight_sums[key] += w

                round_transmit_bytes += float(tensor.numel() * tensor.element_size())

            del state
            torch.cuda.empty_cache()

    aggregated = {}
    if prev_global_params is not None:
        aggregated.update({
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in prev_global_params.items()
        })

    for key, summed in accum.items():
        wsum = weight_sums.get(key, 0.0)
        if wsum > 0.0:
            aggregated[key] = summed / wsum
        else:
            aggregated[key] = summed

    TRAFFIC_STATS["FedHeLLo"]["transmit_MB"] += round_transmit_bytes / MB
    logging.info(
        "[FedHeLLo][epoch %d] round_transmit_MB=%.3f trained_params=%d total_layers=%d",
        epoch,
        round_transmit_bytes / MB,
        len(accum),
        total_layers,
    )
    return aggregated

def truncate(selected_clients_set, output_dir, local_dataset_len_dict, epoch, handle_alpha = False):

    weights_array = torch.tensor(
        [local_dataset_len_dict[client_id] for client_id in selected_clients_set], dtype=torch.float32
    )
    weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)
    weighted_single_weights = {}
    with torch.no_grad():
        for k, client_id in tqdm(enumerate(selected_clients_set)):
            single_output_dir = os.path.join(output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin")
            single_weights = torch.load(single_output_dir, map_location='cpu')
            for key in list(single_weights.keys()):
                if 'local' in key and 'bias' not in key:
                    if 'lora_A' in key:
                        B_key = key.replace('lora_A', 'lora_B')
                        rank = single_weights[B_key].shape[1]
                        if handle_alpha:
                            merge_rate = 16 / rank
                        else:
                            merge_rate = 1
                        if key not in weighted_single_weights.keys():
                            weighted_single_weights[key] = torch.zeros(360, int(single_weights[key].shape[1])).to('cpu')
                            weighted_single_weights[B_key] = torch.zeros(int(single_weights[B_key].shape[0]), 360).to('cpu')
                        weighted_single_weights[key][:rank, :] += single_weights[key] * weights_array[k] * np.sqrt(merge_rate)
                        weighted_single_weights[B_key][:, :rank] += single_weights[B_key] * weights_array[k] * np.sqrt(merge_rate)
                        torch.cuda.empty_cache()
            del single_weights
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return weighted_single_weights

def FlexLoRA(selected_clients_set, output_dir, local_dataset_len_dict, epoch, client_budgets=None, layer_specs=None):
    """
    Aggregate heterogeneous LoRA adapters from clients.

    In addition to the merged weights, this function also updates
    global TRAFFIC_STATS with the communication/compute volume for
    this round. In FlexLoRA, transmit and compute data are the same
    because clients train on all transmitted ranks.
    """
    weights_array = torch.tensor(
        [local_dataset_len_dict[client_id] for client_id in selected_clients_set], dtype=torch.float32
    )
    weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)
    weighted_single_weights = {}

    round_transmit_bytes = 0.0
    round_compute_bytes = 0.0
    compute_ms_per_param = 1.7e-04
    MB = 1024.0 * 1024.0

    with torch.no_grad():
        for k, client_id in tqdm(enumerate(selected_clients_set)):
            comm_bytes = 0.0
            comp_time_ms = 0.0
            single_output_dir = os.path.join(
                output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin"
            )
            single_weights = torch.load(single_output_dir, map_location='cpu')
            for key in list(single_weights.keys()):
                if 'local' in key and 'bias' not in key and 'lora_A' in key:
                    B_key = key.replace('lora_A', 'lora_B')
                    rank = single_weights[B_key].shape[1]
                    merge_rate = 16 / max(rank, 1)
                    new_key = '.'.join(key.split('.')[:-3]) + '.lora'
                    if new_key not in weighted_single_weights.keys():
                        weighted_single_weights[new_key] = 0
                    merged_weight = (single_weights[B_key] @ single_weights[key]) * merge_rate * weights_array[k]
                    weighted_single_weights[new_key] += merged_weight

                    # Track per-round communication/compute volume for this LoRA pair.
                    d_in = single_weights[key].shape[1]
                    d_out = single_weights[B_key].shape[0]
                    elem_bytes = single_weights[key].element_size()
                    bytes_this = (d_in + d_out) * rank * elem_bytes
                    round_transmit_bytes += bytes_this
                    round_compute_bytes += bytes_this  # same for FlexLoRA
                    # Track per-client comm/comp usage
                    base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                    if layer_specs is not None and base_key in layer_specs:
                        spec = layer_specs[base_key]
                        d_out = int(spec.get("d_out", d_out))
                        d_in = int(spec.get("d_in", d_in))
                    comm_bytes += float(rank * (d_in + d_out) * 2.0)
                    comp_time_ms += float(rank * (d_in + d_out) * compute_ms_per_param)

                    del merged_weight
                    torch.cuda.empty_cache()
            del single_weights
            # gc.collect()
            torch.cuda.empty_cache()

            if client_budgets is not None:
                budget = client_budgets.get(int(client_id)) if isinstance(client_budgets, dict) else None
                if budget is not None:
                    B_down_bytes = float(budget.get("B_down_MB", 0.0)) * MB
                    T_ms = float(budget.get("step_ms", 0.0))
                    comm_util = (comm_bytes / B_down_bytes) if B_down_bytes > 0 else 0.0
                    comp_util = (comp_time_ms / T_ms) if T_ms > 0 else 0.0
                    logging.info(
                        "[FlexLoRA][epoch %d][client %s] Comm Util: %.1f%%, Comp Util: %.1f%%",
                        epoch,
                        str(client_id),
                        comm_util * 100.0,
                        comp_util * 100.0,
                    )

    TRAFFIC_STATS["FlexLoRA"]["transmit_MB"] += round_transmit_bytes / MB
    TRAFFIC_STATS["FlexLoRA"]["compute_MB"] += round_compute_bytes / MB
    logging.info(
        "[FlexLoRA][epoch %d] transmit_MB=%.3f compute_MB=%.3f",
        epoch,
        round_transmit_bytes / MB,
        round_compute_bytes / MB,
    )

    torch.cuda.empty_cache()
    return weighted_single_weights



import json
from .rank_allocator import allocate_r_tot_for_client, allocate_r_main_for_client
def _fedhera_dense_global_path(output_dir: str, epoch: int) -> str:
    return os.path.join(output_dir, f"fedhera_dense_global_epoch_{epoch}.pt")

def _load_fedhera_dense_global(output_dir: str, epoch: int):
    path = _fedhera_dense_global_path(output_dir, epoch)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None

def _save_fedhera_dense_global(output_dir: str, epoch: int, dense_dict: dict):
    path = _fedhera_dense_global_path(output_dir, epoch)
    torch.save({k: v.detach().cpu() for k, v in dense_dict.items()}, path)


def FedHera(selected_clients_set, output_dir, local_dataset_len_dict, epoch,
            client_budgets,
            layer_specs,
            quant_scheme=("fp16", "nf4"),
            use_gpu_svd=False,
            basis_update_every=5,
            fixed_client_ranks=None,
            ablation=None,
            lora_alpha=16,
            use_atw=False,
            atw_temperature=2.0, 
            all_client_ids=None, 
            server_agg: str = "original", 
            prev_global_params=None):
    """
    Fed-Hera aggregation:
    1) Merge client adapters into W_global.
    2) [ATW] Compute alignment scores s_i and update cache.
    3) Run SVD to refresh basis.
    4) Allocate r_tot/r_main per client.
    5) [ATW] Calculate lambda and Push truncated A/B plus meta back to clients.
    """
    compute_ms_per_param = 1.7e-04
    
    def _uniform_allocation(layers, bytes_per_col, c_mem_per_col, c_time_per_col,
                            B_down_bytes, M_bytes, T_ms, target_rank=None):
        total_bytes = max(sum(bytes_per_col.values()), 1)
        uniform_cap = B_down_bytes // total_bytes
        if target_rank is not None and target_rank > 0:
            uniform_cap = min(uniform_cap, target_rank)
        r_tot = {L: int(min(uniform_cap, len(meta["sigma"]))) for L, meta in layers.items()}
        mem_denom = max(sum(c_mem_per_col.values()), 1)
        time_denom = max(sum(c_time_per_col.values()), 1)
        r_main_cap = int(min(uniform_cap, M_bytes // mem_denom, T_ms // time_denom))
        r_main = {L: int(min(rt, r_main_cap)) for L, rt in r_tot.items()}
        return r_tot, r_main

    def _random_allocation(layers, bytes_per_col, c_mem_per_col, c_time_per_col,
                           B_down_bytes, M_bytes, T_ms, target_rank=None, rng=None):
        rng = rng or np.random.default_rng()
        caps = {
            L: int(min(len(meta["sigma"]), target_rank)) if target_rank else int(len(meta["sigma"]))
            for L, meta in layers.items()
        }
        r_tot = {L: 0 for L in layers}
        remaining = B_down_bytes
        layer_keys = list(layers.keys())
        while True:
            candidates = [k for k in layer_keys if r_tot[k] < caps[k] and remaining >= bytes_per_col[k]]
            if not candidates:
                break
            choice = rng.choice(candidates)
            r_tot[choice] += 1
            remaining -= bytes_per_col[choice]
        mem_denom = max(sum(c_mem_per_col.values()), 1)
        time_denom = max(sum(c_time_per_col.values()), 1)
        r_main_cap = int(min(M_bytes // mem_denom, T_ms // time_denom))
        r_main = {L: int(min(rt, r_main_cap)) for L, rt in r_tot.items()}
        return r_tot, r_main

    weights_array = torch.tensor([local_dataset_len_dict[c] for c in selected_clients_set], dtype=torch.float32)
    weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)

    # 1. Aggregation Phase
    server_agg_mode = str(server_agg or "original").lower()
    if server_agg_mode not in ["original", "unbiased"]:
        raise ValueError(f"Unsupported FedHera server_agg mode: {server_agg_mode}")

    prev_dense = None
    if server_agg_mode == "unbiased":
        if prev_global_params is not None:
            prev_dense = prev_global_params
        elif epoch > 0:
            prev_dense = _load_fedhera_dense_global(output_dir, epoch - 1)
            if prev_dense is None:
                logging.warning(
                    "[FedHera][epoch %d] Missing dense cache for epoch %d; fallback to original.",
                    epoch, epoch - 1
                )
                server_agg_mode = "original"
        else:
            logging.info("[FedHera][epoch 0] unbiased requires prev dense cache; fallback to original.")
            server_agg_mode = "original"

    if server_agg_mode == "unbiased":
        assert prev_dense is not None, "prev_dense must exist in unbiased mode"

    with torch.no_grad():
        if server_agg_mode == "original":
            aggregated = {}
            for k, client_id in tqdm(enumerate(selected_clients_set)):
                local_path = os.path.join(output_dir, str(client_id),
                                        f"local_output_epoch_{epoch}", "pytorch_model.bin")
                state = torch.load(local_path, map_location="cpu")

                for key in list(state.keys()):
                    if ('local' in key) and ('bias' not in key) and ('lora_A' in key):
                        B_key = key.replace('lora_A', 'lora_B')
                        rank = int(state[B_key].shape[1])
                        merge_rate = float(lora_alpha) / float(max(rank, 1))
                        base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                        merged = (state[B_key] @ state[key]) * merge_rate * weights_array[k]
                        aggregated[base_key] = aggregated.get(base_key, 0) + merged.to("cpu")

                del state
                torch.cuda.empty_cache()

        else:
            delta_sum = {}
            for k, client_id in tqdm(enumerate(selected_clients_set)):
                local_path = os.path.join(output_dir, str(client_id),
                                        f"local_output_epoch_{epoch}", "pytorch_model.bin")
                init_path = os.path.join(output_dir, str(client_id),
                                        f"server_push_epoch_{epoch - 1}", "pytorch_model.bin")

                local_state = torch.load(local_path, map_location="cpu")
                init_state = torch.load(init_path, map_location="cpu") if os.path.exists(init_path) else {}

                for key in list(local_state.keys()):
                    if ('local' in key) and ('bias' not in key) and ('lora_A' in key):
                        B_key = key.replace('lora_A', 'lora_B')
                        rank = int(local_state[B_key].shape[1])
                        merge_rate = float(lora_alpha) / float(max(rank, 1))
                        base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                        merged_local = (local_state[B_key] @ local_state[key]) * merge_rate

                        A_init_key = base_key + "_A.local.weight"
                        B_init_key = base_key + "_B.local.weight"
                        if (A_init_key in init_state) and (B_init_key in init_state):
                            r_init = int(init_state[B_init_key].shape[1])
                            merge_rate_init = float(lora_alpha) / float(max(r_init, 1))
                            merged_init = (init_state[B_init_key] @ init_state[A_init_key]) * merge_rate_init
                        else:
                            merged_init = torch.zeros_like(merged_local)

                        delta = (merged_local - merged_init) * weights_array[k]
                        delta_sum[base_key] = delta_sum.get(base_key, 0) + delta.to("cpu")

                del local_state, init_state
                torch.cuda.empty_cache()

            aggregated = {k: v.clone().to("cpu") for k, v in prev_dense.items()}
            for base_key, d in delta_sum.items():
                aggregated[base_key] = aggregated.get(base_key, 0) + d


    # 2. [ATW Logic] Compute s_i and update cache
    if use_atw:
        logging.info("[FedHera] Computing ATW alignment scores with Trace Optimization...")

        global_sq_norm = 0.0
        for g_tensor in aggregated.values():
            global_sq_norm += torch.linalg.norm(g_tensor.float()) ** 2
        global_norm = math.sqrt(global_sq_norm)

        push_client_ids = all_client_ids if all_client_ids is not None else selected_clients_set
        for client_id in push_client_ids:
            single_output = os.path.join(output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin")
            if not os.path.exists(single_output):
                continue
                
            state = torch.load(single_output, map_location="cpu")
            dot_product = 0.0
            client_sq_norm = 0.0
            
            for base_key, g_tensor in aggregated.items():
                prefix = base_key.rsplit('.lora', 1)[0]
                key_A = None
                key_B = None
                
                for k in state.keys():
                    if prefix in k and 'lora_A' in k:
                        key_A = k
                        key_B = k.replace('lora_A', 'lora_B')
                        break
                
                if key_A and key_B:
                    A_mat = state[key_A].float() # (r, k)
                    B_mat = state[key_B].float() # (d, r)
                    G_mat = g_tensor.float()     # (d, k)
                    
                    rank = B_mat.shape[1]
                    merge_rate = 16 / max(rank, 1)
                    
                    temp_res = B_mat.T @ G_mat 
                    contribution = torch.sum(A_mat * temp_res).item()
                    dot_product += contribution * merge_rate
                    
                    BT_B = B_mat.T @ B_mat   # (r, r)
                    A_AT = A_mat @ A_mat.T   # (r, r)
                    trace_norm = torch.sum(BT_B * A_AT).item()
                    
                    client_sq_norm += trace_norm * (merge_rate ** 2)

            client_norm = math.sqrt(client_sq_norm)
            
            if global_norm > 1e-6 and client_norm > 1e-6:
                s_i = dot_product / (global_norm * client_norm)
            else:
                s_i = 0.0
            
            FEDHERA_CLIENT_STATS[int(client_id)] = {"s": s_i, "t": epoch}
            del state

    # 3. SVD Phase
    basis_version = epoch // max(basis_update_every, 1)
    per_layer_USV = {}
    for layer_key in list(aggregated.keys()):
        Wg = aggregated[layer_key]
        device = "cuda" if (use_gpu_svd and torch.cuda.is_available()) else "cpu"
        W = Wg.to(device=device, dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        per_layer_USV[layer_key] = {
            "U": U.to("cpu"),
            "S": S.to("cpu"),
            "Vh": Vh.to("cpu"),
            "sigma": S.detach().cpu().numpy(),
        }
        del Wg, W, U, S, Vh
        torch.cuda.empty_cache()

    gc.collect()

    # 4. Allocation & Push Phase
    quant_main, quant_res = quant_scheme
    BYTE_MAP = {"fp16": 2, "bfloat16": 2, "int8": 1, "nf4": 0.5, "int4": 0.5}
    bytes_down_main = BYTE_MAP.get(quant_main, 2.0)
    MB = 1024.0 * 1024.0
    rng = np.random.default_rng()
    round_transmit_bytes = 0.0
    round_compute_bytes = 0.0

    push_target_ids = all_client_ids if all_client_ids is not None else selected_clients_set
    for client_id in push_target_ids:
        budgets = client_budgets[int(client_id)]
        B_down_bytes = int(budgets["B_down_MB"] * MB)
        M_bytes = int(budgets["VRAM_MB"] * MB)
        T_ms = float(budgets["step_ms"])
        target_rank = None
        if fixed_client_ranks is not None:
            target_rank = int(fixed_client_ranks.get(int(client_id), 0))
            if target_rank < 0:
                target_rank = 0

        layers = {}
        bytes_per_col = {}
        c_mem_per_col = {}
        c_time_per_col = {}
        for layer_key, usv in per_layer_USV.items():
            if layer_specs is not None and layer_key in layer_specs:
                spec = layer_specs[layer_key]
                d_out, d_in = spec["d_out"], spec["d_in"]
            else:
                d_out = int(usv["U"].shape[0])
                d_in = int(usv["Vh"].shape[1])
            sigma = usv["sigma"]
            layers[layer_key] = {"sigma": sigma, "d_out": d_out, "d_in": d_in}
            bytes_per_col[layer_key] = int((d_out + d_in) * bytes_down_main)
            c_mem_per_col[layer_key] = int((d_out + d_in) * 2.0 * 3.5)
            compute_ms_per_param = 1.7e-04  
            c_time_per_col[layer_key] = float((d_out + d_in) * compute_ms_per_param)

        if ablation == "uniform":
            r_tot, r_main = _uniform_allocation(layers, bytes_per_col, c_mem_per_col, c_time_per_col,
                                                B_down_bytes, M_bytes, T_ms, target_rank)
        elif ablation == "random":
            r_tot, r_main = _random_allocation(layers, bytes_per_col, c_mem_per_col, c_time_per_col,
                                               B_down_bytes, M_bytes, T_ms, target_rank, rng)
        else:
            if target_rank is not None and target_rank > 0:
                r_tot = {L: min(target_rank, len(meta["sigma"])) for L, meta in layers.items()}
            else:
                r_tot, _ = allocate_r_tot_for_client(layers, B_down_bytes, bytes_per_col)
            r_main, _, _ = allocate_r_main_for_client(layers, r_tot, M_bytes, T_ms, c_mem_per_col, c_time_per_col)

        # [ATW Logic] Calculate Lambda for this client
        lambda_val = 1.0
        if use_atw:
            stat = FEDHERA_CLIENT_STATS.get(int(client_id), {"s": 0.0, "t": -1})
            s_stored = stat["s"]
            t_hat = stat["t"]
            
            beta = 0.9
            
            
            current_round = epoch + 1
            decay = beta ** (epoch - t_hat)
            
            exponent = -1.0 * (1.0 + s_stored * decay) * current_round / atw_temperature
            lambda_val = 1.0 - math.exp(exponent)
            
            lambda_val = max(0.0, min(1.0, lambda_val))
            
            if client_id == sorted(list(selected_clients_set))[0]:
                logging.info(f"[ATW] Client {client_id}: s={s_stored:.4f}, lambda={lambda_val:.4f}")

        push_dir = os.path.join(output_dir, str(client_id), f"server_push_epoch_{epoch}")
        os.makedirs(push_dir, exist_ok=True)
        pkg = {}
        meta = {}
        ablation_mode = ablation if ablation is not None else "water_filling"

        for layer_key, usv in per_layer_USV.items():
            rt = int(r_tot.get(layer_key, 0))
            if rt <= 0:
                meta[layer_key] = {"skip": True, "basis_version": basis_version, "r_tot": 0, "r_main": 0,
                                   "ablation": ablation_mode}
                continue
            
            U = usv["U"][:, :rt]
            S = usv["S"][:rt]
            Vh = usv["Vh"][:rt, :]
            
            scale_up = float(rt) / float(max(lora_alpha, 1e-12))
            S = S * scale_up
            sroot = torch.sqrt(S)
            B = (U * sroot.unsqueeze(0))
            A = (sroot.unsqueeze(1) * Vh)
            
            Akey = layer_key + "_A.local.weight"
            Bkey = layer_key + "_B.local.weight"
            pkg[Akey] = A.float().cpu()
            pkg[Bkey] = B.float().cpu()
            
            meta[layer_key] = {
                "skip": False,
                "basis_version": basis_version,
                "r_tot": rt,
                "r_main": int(r_main.get(layer_key, 0)),
                "quant_main": quant_main,
                "quant_res": quant_res,
                "ablation": ablation_mode,
                "lambda": lambda_val
            }
            del U, S, Vh, B, A
        
        torch.save(pkg, os.path.join(push_dir, "pytorch_model.bin"))
        with open(os.path.join(push_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        torch.cuda.empty_cache()

        client_transmit_bytes = 0.0
        client_compute_bytes = 0.0
        comp_time_ms = 0.0
        rank_summary = {}
        for layer_key, info in meta.items():
            if info.get("skip", False):
                continue
            rt = int(info.get("r_tot", 0))
            rm = int(info.get("r_main", 0))
            rank_summary[layer_key] = {"r_tot": rt, "r_main": rm}

            d_out = d_in = None
            if layer_specs and layer_key in layer_specs:
                spec = layer_specs[layer_key]
                d_out = spec.get("d_out")
                d_in = spec.get("d_in")
            if (d_out is None or d_in is None) and pkg:
                Akey = layer_key + "_A.local.weight"
                Bkey = layer_key + "_B.local.weight"
                if d_out is None and Bkey in pkg:
                    d_out = int(pkg[Bkey].shape[0])
                if d_in is None and Akey in pkg:
                    d_in = int(pkg[Akey].shape[1])
            if d_out is None or d_in is None:
                continue

            elem_bytes = bytes_down_main
            bytes_per_rank = (d_out + d_in) * elem_bytes
            client_transmit_bytes += bytes_per_rank * rt
            client_compute_bytes += bytes_per_rank * rm
            comp_time_ms += float(rm * (d_in + d_out) * compute_ms_per_param)

        if rank_summary:
            rank_summary_top = dict(list(rank_summary.items())[:2])
            logging.info("[FedHera][epoch %d][client %s] ranks(top)=%s", epoch, str(client_id), rank_summary_top)
        comm_util = (client_transmit_bytes / float(B_down_bytes)) if B_down_bytes > 0 else 0.0
        comp_util = (comp_time_ms / float(T_ms)) if T_ms > 0 else 0.0
        logging.info(
            "[FedHera][epoch %d][client %s] Comm Util: %.1f%%, Comp Util: %.1f%%",
            epoch,
            str(client_id),
            comm_util * 100.0,
            comp_util * 100.0,
        )
        round_transmit_bytes += client_transmit_bytes
        round_compute_bytes += client_compute_bytes

    TRAFFIC_STATS["FedHera"]["transmit_MB"] += round_transmit_bytes / MB
    TRAFFIC_STATS["FedHera"]["compute_MB"] += round_compute_bytes / MB
    logging.info(
        "[FedHera][epoch %d] round_transmit_MB=%.3f round_compute_MB=%.3f",
        epoch,
        round_transmit_bytes / MB,
        round_compute_bytes / MB,
    )
    aggregated = {k: v.detach().cpu() for k, v in aggregated.items()}
    _save_fedhera_dense_global(output_dir, epoch, aggregated)
    return aggregated


def FedHL(selected_clients_set, output_dir, local_dataset_len_dict, epoch, prev_global_params, layer_specs=None):
    """
    FedHL: Federated Learning for Heterogeneous LoRA via Unbiased Aggregation.
    Paper: arXiv:2505.18494v1

    Formula: W_{t+1} = W_t + sum( p_i * (W_{client_i} - W_t^{rank_i}) )
    Where p_i is optimized based on truncation error.
    """
    logging.info(f"[FedHL] Starting aggregation for epoch {epoch}")
    
    layer_svd_cache = {}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    client_ranks = {} 
    client_errors = {} 
    client_updates_cache = {} 
    
    epsilon = 1e-6
    
    for client_id in tqdm(selected_clients_set, desc="[FedHL] Computing Errors"):
        single_output_dir = os.path.join(output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin")
        if not os.path.exists(single_output_dir):
            continue
            
        single_weights = torch.load(single_output_dir, map_location='cpu')
        client_updates_cache[client_id] = single_weights
        
        current_client_ranks = {}
        total_error = 0.0
        
        for key in single_weights.keys():
            if 'local' in key and 'lora_A' in key:
                base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                B_key = key.replace('lora_A', 'lora_B')
                
                rank = single_weights[key].shape[0] # A is [r, d_in]
                current_client_ranks[base_key] = rank
                
                if base_key not in prev_global_params:
                    continue
                
                if base_key not in layer_svd_cache:
                    W_t_layer = prev_global_params[base_key].to(device).float()
                    U, S, Vh = torch.linalg.svd(W_t_layer, full_matrices=False)
                    layer_svd_cache[base_key] = (U.cpu(), S.cpu(), Vh.cpu())
                    del W_t_layer
                    torch.cuda.empty_cache()
                
                _, S, _ = layer_svd_cache[base_key]
                truncated_singular_values = S[rank:]
                layer_error = torch.sum(truncated_singular_values ** 2).item()
                total_error += layer_error
                
        client_ranks[client_id] = current_client_ranks
        client_errors[client_id] = total_error
    
    p_numerators = {}
    p_denom_sum = 0.0
    
    for cid in selected_clients_set:
        err = client_errors.get(cid, 0.0)
        val = 1.0 / ( (err ** 2) + epsilon )
        p_numerators[cid] = val
        p_denom_sum += val
        
    optimal_weights = {cid: val / p_denom_sum for cid, val in p_numerators.items()}
    
    log_weights = {cid: f"{w:.4f}" for cid, w in optimal_weights.items()}
    logging.info(f"[FedHL] Aggregation Weights: {str(log_weights)}")

    global_delta = {k: torch.zeros_like(v, device='cpu') for k, v in prev_global_params.items()}
    
    for client_id in tqdm(selected_clients_set, desc="[FedHL] Aggregating"):
        weight = optimal_weights[client_id]
        single_weights = client_updates_cache[client_id]
        ranks = client_ranks.get(client_id, {})
        
        for key in single_weights.keys():
            if 'local' in key and 'lora_A' in key:
                base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                B_key = key.replace('lora_A', 'lora_B')
                
                if base_key not in layer_svd_cache:
                    continue
                    
                B_mat = single_weights[B_key].float()
                A_mat = single_weights[key].float()
                
                rank = ranks[base_key]
                merge_rate = 16.0 / max(rank, 1) 
                
                W_client = (B_mat @ A_mat) * merge_rate
                
                U, S, Vh = layer_svd_cache[base_key]
                U_r = U[:, :rank]
                S_r = S[:rank]
                Vh_r = Vh[:rank, :]
                
                W_truncated = (U_r @ torch.diag(S_r) @ Vh_r)
                
                diff = (W_client - W_truncated) * weight
                
                if base_key in global_delta:
                    global_delta[base_key] += diff
                
    new_global_params = {}
    for k, v in prev_global_params.items():
        if k in global_delta:
            updated_tensor = (v + global_delta[k]).detach().clone()
            new_global_params[k] = updated_tensor
        else:
            new_global_params[k] = v.detach().clone()
            
    del layer_svd_cache
    del client_updates_cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return new_global_params
