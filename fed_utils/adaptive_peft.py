import random
import numpy as np
import os
import torch
import peft
from tqdm import tqdm

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def tokenize(tokenizer, prompt, cutoff_len=512, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def load_weight_local(weighted_single_weights, model):
    weight_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(param.shape)
            print(name)
            rank = min(param.shape[0], param.shape[1])
            if name + '.' + str(rank) in weighted_single_weights.keys():
                weight_dict[name] = weighted_single_weights[name + '.' + str(rank)]
    return weight_dict

def load_weight_SLoRA(weighted_single_weights, model):
    weight_dict = {}
    with torch.no_grad():
        for key, val in weighted_single_weights.items():
            for name, param in model.named_parameters():
                if key == name:
                    weight_dict[key] = val.to(param.data.device) - param

        model_tune_param = {name: param for name, param in model.named_parameters() if param.requires_grad}
        for key, val in weight_dict.items():
            key_check = '.'.join(key.split('.')[:-1])
            for name, param in model_tune_param.items():
                if key_check in name and 'lora_A' in name:
                    rank = min(param.shape)
                    merge_rate = 16/rank
                    W_cpu = weight_dict[key].detach().to('cpu')
                    u, s, vT = torch.linalg.svd(W_cpu, full_matrices=False)
                    u = u[:, :rank]
                    s = s[:rank]
                    v = vT[:rank, :]
                    lora_B = u @ torch.diag(s)/merge_rate
                    lora_A = v
                    model_tune_param[name] = lora_A
                    B_name = name.replace('lora_A', 'lora_B')
                    model_tune_param[B_name] = lora_B
    return model_tune_param

def distribute_weight(weighted_single_weights, model):
    # mode is local model
    # around 15 min for one client
    weight_dict = {}
    for key in tqdm(weighted_single_weights.keys()):
        # _, target, target_name = peft.utils.other._get_submodules(model, key + '_A.local')
        rank = 2048
        merge_rate = 16 / rank
        W_cpu = (weighted_single_weights[key] / merge_rate).detach().to('cpu')
        u, s, vT = torch.linalg.svd(W_cpu, full_matrices=False)
        u = u[:, :rank]
        s = s[:rank]
        v = vT[:rank, :]
        lora_B = u @ torch.diag(s)
        lora_A = v
        weight_dict[key + '_A.local.weight'] = lora_A
        weight_dict[key + '_B.local.weight'] = lora_B
        # print(key + '_A.local.weight', lora_A.shape)
        # print(key + '_B.local.weight', lora_B.shape)
    return weight_dict

def distribute_weight_fast(weighted_single_weights, config_local):
    # mode is local model, model needs to load local weights first
    weight_dict = {}
    rank_dict = {}
    alpha = config_local['alpha']
    for client, val in config_local.items():
        if 'Client' in client:
            for key in val.keys():
                if key in rank_dict.keys():
                    rank_dict[key].append(val[key])
                else:
                    rank_dict[key] = [val[key]]

    for key in tqdm(weighted_single_weights.keys()):
        W_cpu = weighted_single_weights[key].detach().to('cpu')
        u, s, vT = torch.linalg.svd(W_cpu, full_matrices=False)
        for layer, rank_lst in rank_dict.items():
            if layer in key:
                break
        for rank in rank_lst:
            if rank != 0:
                U = u[:, :rank]
                S = s[:rank]
                V = vT[:rank, :]
                lora_B = U @ torch.diag(S)
                lora_A = V
                # merge_rate = 2
                merge_rate = alpha/rank
                weight_dict[key + '_A.local.weight.' + str(rank)] = lora_A
                weight_dict[key + '_B.local.weight.' + str(rank)] = lora_B/ merge_rate
    return weight_dict


def modify_adapter(peft_model, adapter_name, modify_module_rank ={},layer_dict = [], lora_alpha = 16, lora_dropout = 0.05, init_lora_weights = True):
    for name, module in peft_model.named_modules():
        if any(['.' + str(layer) + '.' in name for layer in layer_dict]):
            for key, r in modify_module_rank.items():
                if lora_alpha == 0:
                    alpha = r
                else:
                    alpha = lora_alpha
                if key in name and isinstance(module, peft.tuners.lora.Linear):
                    module.update_layer(adapter_name, r, alpha, lora_dropout, init_lora_weights)
                if key in name and isinstance(module, peft.tuners.lora.Linear8bitLt):
                    module.update_layer(adapter_name, r, alpha, lora_dropout, init_lora_weights)


# fed_utils/adaptive_peft.py (append)
import json
import torch.nn as nn

def apply_lora_prefix_mask(peft_model, per_layer_r_main):
    """
    peft_model: PEFT LoRA 模型
    per_layer_r_main: Dict[layer_key] -> int
    对 A/B 注册梯度hook：只让前 r_main 列/行产生梯度。
    """
    hooks = []
    for name, param in peft_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            # name 示例: model.layers.0.self_attn.q_proj.lora_A.local.weight
            base_key = '.'.join(name.split('.')[:-3]) + '.lora'
            r_main = int(per_layer_r_main.get(base_key, 0))
            if r_main <= 0:
                mask = torch.zeros_like(param, dtype=param.dtype, device=param.device)
            else:
                if "lora_A" in name:
                    # A: [r, d_in] -> 只保留前 r_main 行
                    mask = torch.zeros_like(param)
                    mask[:r_main, :] = 1
                else:
                    # B: [d_out, r] -> 只保留前 r_main 列
                    mask = torch.zeros_like(param)
                    mask[:, :r_main] = 1

            def _make_hook(msk):
                def hook_fn(grad):
                    if grad is None: return None
                    return grad * msk.to(grad.device)
                return hook_fn

            hooks.append(param.register_hook(_make_hook(mask)))
    return hooks

def load_weight_fedhera_if_exists(output_dir, client_id, epoch):
    """
    若存在 server_push 包，读取并返回 (state_dict, meta)；否则返回 (None, None)
    """
    import os, json, torch
    push_dir = os.path.join(output_dir, str(client_id), f"server_push_epoch_{epoch}")
    model_path = os.path.join(push_dir, "pytorch_model.bin")
    meta_path  = os.path.join(push_dir, "meta.json")
    if os.path.exists(model_path) and os.path.exists(meta_path):
        state = torch.load(model_path, map_location="cpu")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return state, meta
    return None, None
