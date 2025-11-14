import numpy as np
import torch
import os
from torch.nn.functional import normalize
import gc
from tqdm import tqdm


def FedAvg(selected_clients_set, output_dir, local_dataset_len_dict, epoch):
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
        del single_weights
        gc.collect()
        torch.cuda.empty_cache()

    # set_peft_model_state_dict(model, weighted_single_weights, "default")
    torch.cuda.empty_cache()
    return weighted_single_weights

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

def FlexLoRA(selected_clients_set, output_dir, local_dataset_len_dict, epoch):
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
                        merge_rate = 16 / rank
                        new_key = '.'.join(key.split('.')[:-3]) + '.lora'
                        if new_key not in weighted_single_weights.keys():
                            weighted_single_weights[new_key] = 0
                        merged_weight = (single_weights[B_key] @ single_weights[key]) * merge_rate * weights_array[k]
                        weighted_single_weights[new_key] += merged_weight
                        del merged_weight
                        torch.cuda.empty_cache()
            del single_weights
            # gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return weighted_single_weights


# fed_utils/model_aggregation.py (append)

import json
from .rank_allocator import allocate_r_tot_for_client, allocate_r_main_for_client

def FedHera(selected_clients_set, output_dir, local_dataset_len_dict, epoch,
            client_budgets,                 # Dict[client_id] -> {"B_down_MB","VRAM_MB","step_ms"}
            layer_specs,                    # Dict[layer_key] -> {"d_out","d_in"}
            quant_scheme=("fp16","nf4"),    # (quant_main, quant_res) 控制字节估算
            use_gpu_svd=False,              # 可选：GPU 上做SVD
            basis_update_every=5,           # 每 K 轮更新一次基底
            ):
    """
    返回聚合后的 "全局Wg"（便于日志/可视化），并在磁盘上为每客户端写入 server_push 包。
    """
    # 1) 读入各客户端本轮 LoRA，合成 Wg（与 FlexLoRA 类似）
    weights_array = torch.tensor([local_dataset_len_dict[c] for c in selected_clients_set], dtype=torch.float32)
    weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)

    with torch.no_grad():
        aggregated = {}  # key(".lora") -> Tensor [d_out, d_in]
        for k, client_id in tqdm(enumerate(selected_clients_set)):
            single_output = os.path.join(output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin")
            state = torch.load(single_output, map_location="cpu")
            for key in list(state.keys()):
                if 'local' in key and 'bias' not in key and ('lora_A' in key):
                    B_key = key.replace('lora_A', 'lora_B')
                    rank = state[B_key].shape[1]
                    merge_rate = 16 / rank
                    base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                    merged = (state[B_key] @ state[key]) * merge_rate * weights_array[k]
                    if base_key not in aggregated:
                        aggregated[base_key] = merged.clone().to('cpu')
                    else:
                        aggregated[base_key] += merged.to('cpu')
            del state
            torch.cuda.empty_cache()

    # 2) 是否需要本轮更新基底（可每 K 轮）
    basis_version = epoch // max(basis_update_every, 1)

    # 3) 对每层做 SVD，准备每层奇异值谱与 U,S,V
    per_layer_USV = {}
    for layer_key, Wg in aggregated.items():
        device = "cuda" if (use_gpu_svd and torch.cuda.is_available()) else "cpu"
        W = Wg.to(device=device, dtype=torch.float32)  # 用 fp32 做 SVD 更稳定
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        # 为节省下行，先回 CPU
        per_layer_USV[layer_key] = {
            "U": U.to("cpu"),
            "S": S.to("cpu"),
            "Vh": Vh.to("cpu"),
            "sigma": S.detach().cpu().numpy()
        }
        del U, S, Vh, W
        torch.cuda.empty_cache()

    # 4) 为每客户端做 rank 分配并写入 server_push 包
    quant_main, quant_res = quant_scheme
    BYTE_MAP = {"fp16":2, "bfloat16":2, "int8":1, "nf4":0.5, "int4":0.5}
    bytes_down_main = BYTE_MAP.get(quant_main, 2.0)
    bytes_down_res  = BYTE_MAP.get(quant_res, 0.5)
    # 统一用 "每列字节" 估算：(d_out + d_in) * bytes
    # 下发时我们一次性给到 r_tot 列（包含 main+res），bytes 用更保守的主精度估。
    for client_id in selected_clients_set:
        budgets = client_budgets[int(client_id)]
        B_down_bytes = int(budgets["B_down_MB"] * 1024 * 1024)
        M_bytes      = int(budgets["VRAM_MB"]   * 1024 * 1024)
        T_ms         = float(budgets["step_ms"])

        # 准备层元信息
        layers = {}
        bytes_per_col = {}
        c_mem_per_col = {}
        c_time_per_col = {}
        for layer_key, usv in per_layer_USV.items():
            print("="*50)
            print(f"DEBUG: 尝试访问的 key: {layer_key}")
            print(f"DEBUG: layer_specs 中所有可用的 keys: {list(layer_specs.keys())}")
            print("="*50)
            if layer_specs is not None and layer_key in layer_specs:
                spec = layer_specs[layer_key]
                d_out, d_in = spec["d_out"], spec["d_in"]
            else:
                d_out = int(usv["U"].shape[0])
                d_in = int(usv["Vh"].shape[1])
            sigma = usv["sigma"]
            layers[layer_key] = {"sigma": sigma, "d_out":d_out, "d_in":d_in}
            bytes_per_col[layer_key] = int((d_out + d_in) * bytes_down_main)
            # 训练侧开销估算：显存 bytes（考虑优化器多副本）
            c_mem_per_col[layer_key]  = int((d_out + d_in) * 2.0 * 3.5)  # 2B(bf16)×(1参数+优化器状态系数~3.5)
            # 步时线性斜率：给个常数，也可通过 burn-in 标定
            c_time_per_col[layer_key] = 1.0

        # 下载水位
        r_tot, _ = allocate_r_tot_for_client(layers, B_down_bytes, bytes_per_col)
        # 训练水位
        r_main, _, _ = allocate_r_main_for_client(layers, r_tot, M_bytes, T_ms, c_mem_per_col, c_time_per_col)

        # 生成该客户端的下发包：按 r_tot 截取 U,S,V 并合成为 B,A；同时附上 r_main 掩码信息
        push_dir = os.path.join(output_dir, str(client_id), f"server_push_epoch_{epoch}")
        os.makedirs(push_dir, exist_ok=True)
        pkg = {}
        meta = {}
        for layer_key, usv in per_layer_USV.items():
            rt = int(r_tot.get(layer_key, 0))
            if rt <= 0:
                meta[layer_key] = {"skip": True, "basis_version": basis_version, "r_tot": 0, "r_main": 0}
                continue
            U = usv["U"][:, :rt]
            S = usv["S"][:rt]
            Vh= usv["Vh"][:rt, :]
            # LoRA友好因子：B = U * sqrt(S), A = sqrt(S) * V^T
            sroot = torch.sqrt(S)
            B = (U * sroot.unsqueeze(0))        # [d_out, rt]
            A = (sroot.unsqueeze(1) * Vh)       # [rt, d_in]
            # 保存为与本工程一致的 Key 命名
            # 例："...q_proj.lora_A.local.weight" / "...q_proj.lora_B.local.weight"
            Akey = layer_key + "_A.local.weight"
            Bkey = layer_key + "_B.local.weight"
            pkg[Akey] = B.float().cpu()  # 注意：我们用B给_A，Vh给_B 与原 distribute_weight_fast 对齐方式保持一致性
            pkg[Bkey] = A.float().cpu()
            # Correct A/B placement: ensure lora_A gets A and lora_B gets B
            pkg[Akey] = A.float().cpu()
            pkg[Bkey] = B.float().cpu()
            meta[layer_key] = {
                "skip": False, "basis_version": basis_version,
                "r_tot": rt, "r_main": int(r_main.get(layer_key, 0)),
                "quant_main": quant_main, "quant_res": quant_res
            }
            del U, S, Vh, B, A
        torch.save(pkg, os.path.join(push_dir, "pytorch_model.bin"))
        with open(os.path.join(push_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        torch.cuda.empty_cache()

    # 返回聚合的全局 Wg（便于日志/可视化；实际下发已写盘）
    return aggregated
