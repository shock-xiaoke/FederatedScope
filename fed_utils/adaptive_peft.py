import random
import numpy as np
import os
import torch
import peft
from tqdm import tqdm

def seed_torch(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # Let cuDNN/TF32 stay fast by default; flip deterministic on only if requested.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic

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


def distribute_weight(weighted_single_weights, model):
    weight_dict = {}
    for key in tqdm(weighted_single_weights.keys()):
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
        W_cpu = weighted_single_weights[key].detach().to(device='cpu', dtype=torch.float32)
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
                merge_rate = alpha/rank
                weight_dict[key + '_A.local.weight.' + str(rank)] = lora_A
                weight_dict[key + '_B.local.weight.' + str(rank)] = lora_B/ merge_rate
    return weight_dict


def modify_adapter(peft_model, adapter_name, modify_module_rank=None, layer_dict=None,
                   lora_alpha=16, lora_dropout=0.05, init_lora_weights=True):
    """
    Update LoRA ranks for modules whose names contain keys in ``modify_module_rank``.
    If ``layer_dict`` is None or empty, all layers are considered; otherwise only
    layers whose name contains ``.{layer}.`` for some layer in ``layer_dict`` are updated.
    """
    if modify_module_rank is None:
        modify_module_rank = {}
    if layer_dict is None:
        layer_dict = []

    for name, module in peft_model.named_modules():
        if layer_dict and not any(f".{layer}." in name for layer in layer_dict):
            continue
        for key, r in modify_module_rank.items():
            if lora_alpha == 0:
                alpha = r
            else:
                alpha = lora_alpha
            
            if key in name and (isinstance(module, peft.tuners.lora.Linear) or isinstance(module, peft.tuners.lora.Linear8bitLt)):
                use_rslora = False
                try:
                    if hasattr(peft_model, "peft_config") and adapter_name in peft_model.peft_config:
                        use_rslora = bool(getattr(peft_model.peft_config[adapter_name], "use_rslora", False))
                except Exception:
                    use_rslora = False

                try:
                    module.update_layer(adapter_name, r, alpha, lora_dropout, init_lora_weights, use_rslora)
                except TypeError:
                    module.update_layer(adapter_name, r, alpha, lora_dropout, init_lora_weights)


import json
import torch.nn as nn

def apply_lora_prefix_mask(peft_model, per_layer_r_main):
    hooks = []
    for name, param in peft_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            base_key = '.'.join(name.split('.')[:-3]) + '.lora'
            r_main = int(per_layer_r_main.get(base_key, 0))
            if r_main <= 0:
                mask = torch.zeros_like(param, dtype=param.dtype, device=param.device)
            else:
                if "lora_A" in name:
                    mask = torch.zeros_like(param)
                    mask[:r_main, :] = 1
                else:
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
    import os, json, torch
    push_dir = os.path.join(output_dir, str(client_id), f"server_push_epoch_{epoch}")
    model_path = os.path.join(push_dir, "pytorch_model.bin")
    meta_path  = os.path.join(push_dir, "meta.json")
    
    if os.path.exists(model_path) and os.path.exists(meta_path):
        state = torch.load(model_path, map_location="cpu")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        for layer_key, info in meta.items():
            if info.get("skip", False):
                continue
            
            lambda_val = info.get("lambda", 1.0)
            if lambda_val >= 0.999:
                continue
                
            r_main = int(info.get("r_main", 0))
            r_tot = int(info.get("r_tot", 0))
            
            if r_main >= r_tot:
                continue

            key_A = layer_key + "_A.local.weight"
            key_B = layer_key + "_B.local.weight"
            
            if key_A in state and key_B in state:
                tensor_A = state[key_A]
                tensor_B = state[key_B]
                
                current_r_A = tensor_A.shape[0]
                current_r_B = tensor_B.shape[1]
                
                if r_main >= current_r_A or r_main >= current_r_B:
                    continue

                scale = lambda_val ** 0.5
                
                tensor_A[r_main:, :].mul_(scale)
                tensor_B[:, r_main:].mul_(scale)
                
        return state, meta
        
    return None, None
