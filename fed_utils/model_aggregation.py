import numpy as np
import torch
import os
from torch.nn.functional import normalize
import gc
import logging
from tqdm import tqdm


# Simple global traffic counters to let the training loop
# summarise total communication and compute volume.
TRAFFIC_STATS = {
    "FlexLoRA": {"transmit_MB": 0.0, "compute_MB": 0.0},
    "FedHera": {"transmit_MB": 0.0, "compute_MB": 0.0},
    "FedHeLLo": {"transmit_MB": 0.0, "compute_MB": 0.0},
}


def reset_traffic_stats():
    for method in TRAFFIC_STATS:
        TRAFFIC_STATS[method]["transmit_MB"] = 0.0
        TRAFFIC_STATS[method]["compute_MB"] = 0.0


def get_traffic_stats():
    # Return a shallow copy so callers cannot mutate internals.
    return {
        name: stats.copy()
        for name, stats in TRAFFIC_STATS.items()
    }


def FedAvg(selected_clients_set, output_dir, local_dataset_len_dict, epoch, client_budgets=None, layer_specs=None):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)
    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(client_id), "local_output_epoch_{}".format(epoch),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir, map_location='cpu')
        # delete_lst = []
        # for key in single_weights.keys():
        #     if 'bias' in key and 'lora_B' in key:
        #         delete_lst.append(key)
        # for key in delete_lst:
        #     del single_weights[key]
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

    # set_peft_model_state_dict(model, weighted_single_weights, "default")
    torch.cuda.empty_cache()
    return weighted_single_weights

def FedHeLLo(selected_clients_set, output_dir, local_dataset_len_dict, epoch,
             active_layers_map=None, prev_global_params=None, layer_specs=None):
    """
    Layer-wise aggregation for Fed-HeLLo that averages only over clients
    that trained each LoRA layer.
    """
    del local_dataset_len_dict  # unused but kept for API symmetry
    layer_counts = {}
    accum = {}
    round_transmit_bytes = 0.0
    MB = 1024.0 * 1024.0
    total_layers = len(layer_specs or {})

    with torch.no_grad():
        for client_id in selected_clients_set:
            single_output_dir = os.path.join(
                output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin"
            )
            if not os.path.exists(single_output_dir):
                continue
            state = torch.load(single_output_dir, map_location="cpu")
            active_layers = set(active_layers_map.get(client_id, [])) if active_layers_map else None
            updated_layers = set()
            for key, tensor in state.items():
                base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                if active_layers is not None and base_key not in active_layers:
                    continue
                updated_layers.add(base_key)
                if key not in accum:
                    accum[key] = torch.zeros_like(tensor)
                accum[key] += tensor
                round_transmit_bytes += float(tensor.numel() * tensor.element_size())
            for base_key in updated_layers:
                layer_counts[base_key] = layer_counts.get(base_key, 0) + 1
            del state
            torch.cuda.empty_cache()

    aggregated = {}
    if prev_global_params is not None:
        aggregated.update({
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in prev_global_params.items()
        })

    for key, summed in accum.items():
        base_key = '.'.join(key.split('.')[:-3]) + '.lora'
        count = layer_counts.get(base_key, 0)
        if count > 0:
            aggregated[key] = summed / float(count)
        elif prev_global_params is not None and key in prev_global_params:
            aggregated[key] = prev_global_params[key]

    TRAFFIC_STATS["FedHeLLo"]["transmit_MB"] += round_transmit_bytes / MB
    logging.info(
        "[FedHeLLo][epoch %d] round_transmit_MB=%.3f trained_layers=%d/%d",
        epoch,
        round_transmit_bytes / MB,
        len(layer_counts),
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
                        # new_key = '.'.join(key.split('.')[:-3]) + '.lora'
                        if key not in weighted_single_weights.keys():
                            weighted_single_weights[key] = torch.zeros(360, int(single_weights[key].shape[1])).to('cpu')
                            weighted_single_weights[B_key] = torch.zeros(int(single_weights[B_key].shape[0]), 360).to('cpu')
                        weighted_single_weights[key][:rank, :] += single_weights[key] * weights_array[k] * np.sqrt(merge_rate)
                        weighted_single_weights[B_key][:, :rank] += single_weights[B_key] * weights_array[k] * np.sqrt(merge_rate)
                        torch.cuda.empty_cache()
            del single_weights
            # gc.collect()
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


# fed_utils/model_aggregation.py (append)

import json
from .rank_allocator import allocate_r_tot_for_client, allocate_r_main_for_client

def FedHera(selected_clients_set, output_dir, local_dataset_len_dict, epoch,
            client_budgets,
            layer_specs,
            quant_scheme=("fp16", "nf4"),
            use_gpu_svd=False,
            basis_update_every=5,
            fixed_client_ranks=None,
            ablation=None,
            lora_alpha=16):
    """
    Fed-Hera aggregation:
    1) Merge client adapters into W_global.
    2) Run SVD to refresh basis.
    3) Allocate r_tot/r_main per client (water-filling or ablation).
    4) Push truncated A/B plus meta (r_main) back to clients.
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

    with torch.no_grad():
        aggregated = {}
        for k, client_id in tqdm(enumerate(selected_clients_set)):
            single_output = os.path.join(output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin")
            state = torch.load(single_output, map_location="cpu")
            for key in list(state.keys()):
                if 'local' in key and 'bias' not in key and ('lora_A' in key):
                    B_key = key.replace('lora_A', 'lora_B')
                    rank = state[B_key].shape[1]
                    merge_rate = 16 / max(rank, 1)
                    base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                    merged = (state[B_key] @ state[key]) * merge_rate * weights_array[k]
                    if base_key not in aggregated:
                        aggregated[base_key] = merged.clone().to('cpu')
                    else:
                        aggregated[base_key] += merged.to('cpu')
            del state
            torch.cuda.empty_cache()

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

    quant_main, quant_res = quant_scheme
    BYTE_MAP = {"fp16": 2, "bfloat16": 2, "int8": 1, "nf4": 0.5, "int4": 0.5}
    bytes_down_main = BYTE_MAP.get(quant_main, 2.0)
    MB = 1024.0 * 1024.0
    rng = np.random.default_rng()
    round_transmit_bytes = 0.0
    round_compute_bytes = 0.0

    for client_id in selected_clients_set:
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
            compute_ms_per_param = 1.7e-04  # 与 main.py 里的数保持同步
            # 每个 rank 的时间成本 ≈ (d_out + d_in) * compute_ms_per_param
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
            # Pre-scale the singular values so that (B@A)*(alpha/rt) matches W_svd on the client.
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
            }
            del U, S, Vh, B, A
        torch.save(pkg, os.path.join(push_dir, "pytorch_model.bin"))
        with open(os.path.join(push_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        torch.cuda.empty_cache()

        # Track communication/compute stats for this client based on the written meta.
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
            # Fallback to shapes from the stored tensors if specs are missing.
            if (d_out is None or d_in is None) and pkg:
                Akey = layer_key + "_A.local.weight"
                Bkey = layer_key + "_B.local.weight"
                if d_out is None and Bkey in pkg:
                    d_out = int(pkg[Bkey].shape[0])
                if d_in is None and Akey in pkg:
                    d_in = int(pkg[Akey].shape[1])
            if d_out is None or d_in is None:
                continue

            Akey = layer_key + "_A.local.weight"
            Bkey = layer_key + "_B.local.weight"
            # Use the quantized main precision for accounting to match budgets.
            elem_bytes = bytes_down_main

            bytes_per_rank = (d_out + d_in) * elem_bytes
            client_transmit_bytes += bytes_per_rank * rt
            client_compute_bytes += bytes_per_rank * rm
            comp_time_ms += float(rm * (d_in + d_out) * compute_ms_per_param)

        if rank_summary:
            logging.info("[FedHera][epoch %d][client %s] ranks=%s", epoch, str(client_id), rank_summary)
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

    return aggregated

