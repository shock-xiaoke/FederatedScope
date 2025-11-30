from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    # For older transformers versions, register newer model types on the fly.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except Exception:
        LlamaConfig = None
except Exception:
    CONFIG_MAPPING = None
    LlamaConfig = None
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from fed_utils import FedAvg, client_selection, seed_torch, GeneralClient, FlexLoRA, \
    load_weight_local, distribute_weight_fast, modify_adapter, FedHera
from fed_utils.model_aggregation import reset_traffic_stats, get_traffic_stats, TRAFFIC_STATS

import datasets
from datasets import load_dataset
from utils.prompter import Prompter
import socket

datasets.utils.logging.set_verbosity_error()

import numpy as np
import random
import os
import torch
import logging
import argparse
os.environ["WANDB_MODE"]="disabled"

import json

# Canonical LoRA ranks associated with client resource levels.
# These are used both for heterogeneous FlexLoRA ranks and, via
# Fed-Hera, for computing client resource budgets.
RESOURCE_RANKS = {
    "poor": 4,
    "medium": 8,
    "high": 16,
}

# Deterministic client rank map for comparability.
def build_fixed_rank_map(num_clients):
    rank_map = {}
    for i in range(num_clients):
        if 0 <= i <= 4:
            rank = 4
        elif 5 <= i <= 14:
            rank = 8
        elif 15 <= i <= 19:
            rank = 16
        else:
            # Default to the medium tier for any additional clients.
            rank = RESOURCE_RANKS["medium"]
        rank_map[i] = rank
    return rank_map


def get_client_budgets(num_clients, hetero_mode, seed=42):
    """
    Hardcoded per-client resource budgets used for all aggregation methods.
    setting_A is homogeneous and calibrated so B_down~40MB and step_ms~800
    yield roughly r_comm~64 and r_comp~16 on Mistral-7B-scale models.
    """
    rng = np.random.default_rng(seed)
    TIERS = {
        # Communication-light but slower compute: ~64 comm rank, ~16 compute rank (Mistral-7B).
        "bandwidth_rich_compute_poor": {"B_down_MB": 40.0, "VRAM_MB": 48000.0, "step_ms": 1600.0},
        # Balanced/stronger hardware.
        "high_resource": {"B_down_MB": 96.0, "VRAM_MB": 48000.0, "step_ms": 520.0},
        # Plenty of bandwidth, moderate compute.
        "decoupled": {"B_down_MB": 72.0, "VRAM_MB": 32000.0, "step_ms": 720.0},
        # Constrained clients.
        "low_resource": {"B_down_MB": 32.0, "VRAM_MB": 16000.0, "step_ms": 1200.0},
    }

    client_budgets = {}
    if hetero_mode == 'setting_A':
        base = TIERS["bandwidth_rich_compute_poor"]
        for i in range(num_clients):
            client_budgets[i] = {
                "tier": "bandwidth_rich_compute_poor",
                "B_down_MB": float(base["B_down_MB"]),
                "VRAM_MB": float(base["VRAM_MB"]),
                "step_ms": float(base["step_ms"]),
            }
        return client_budgets

    probs = [0.2, 0.5, 0.30]
    tier_names = ["high_resource", "decoupled", "low_resource"]
    for i in range(num_clients):
        tier = str(rng.choice(tier_names, p=probs))
        base = TIERS[tier]
        client_budgets[i] = {
            "tier": tier,
            "B_down_MB": float(base["B_down_MB"]),
            "VRAM_MB": float(base["VRAM_MB"]),
            "step_ms": float(base["step_ms"]),
        }
    return client_budgets

def extract_lora_layer_specs(model):
    """
    Collect LoRA layer (d_out, d_in) pairs keyed by the shared base name.
    """
    layer_specs = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            base_key = '.'.join(name.split('.')[:-3]) + '.lora'
            if base_key not in layer_specs:
                layer_specs[base_key] = {"d_out": None, "d_in": None}
            if "lora_A" in name:
                _, d_in = param.shape
                layer_specs[base_key]["d_in"] = int(d_in)
            else:
                d_out, _ = param.shape
                layer_specs[base_key]["d_out"] = int(d_out)

    # Fill missing dimensions when only one side of the adapter was seen.
    for key, spec in layer_specs.items():
        if spec["d_out"] is None or spec["d_in"] is None:
            for name, param in model.named_parameters():
                if key in name:
                    if spec["d_out"] is None and "lora_B" in name:
                        spec["d_out"] = int(param.shape[0])
                    if spec["d_in"] is None and "lora_A" in name:
                        spec["d_in"] = int(param.shape[1])
    return layer_specs

def calculate_unified_rank_from_budget(client_budgets, layer_specs, max_rank=64):
    """
    Derive per-client ranks from shared communication/computation budgets.
    Unit costs are calibrated so that a setting_A client with B_down~40MB and
    step_ms~800 yields r_comm~64 and r_comp~16 on Mistral-7B-style models.
    """
    rank_map = {}

    total_dim_sum = 0
    for _, spec in layer_specs.items():
        d_in = spec.get("d_in")
        d_out = spec.get("d_out")
        if d_in is None or d_out is None:
            continue
        total_dim_sum += (d_in + d_out)

    if total_dim_sum == 0:
        return {client_id: 8 for client_id in client_budgets.keys()}

    MB = 1024 * 1024
    # NF4-ish download with some overhead to hit the target r_comm.
    bytes_per_param = 0.8
    bytes_per_unit_rank = total_dim_sum * bytes_per_param

    # Approximate compute cost calibrated for Mistral-7B.
    compute_ms_per_param = 6.0e-05
    time_cost_per_unit_rank = total_dim_sum * compute_ms_per_param

    for client_id, budget in client_budgets.items():
        b_down = float(budget.get("B_down_MB", 0.0))
        step_ms = float(budget.get("step_ms", 0.0))

        r_comm = int((b_down * MB) / bytes_per_unit_rank) if bytes_per_unit_rank > 0 else max_rank
        r_comp = int(step_ms / time_cost_per_unit_rank) if time_cost_per_unit_rank > 0 else max_rank

        final_r = max(1, min(max_rank, min(r_comm, r_comp)))
        rank_map[client_id] = final_r

    return rank_map

def parse_lora_target_modules(s):
    # Accept JSON list or comma-separated string
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    return [x.strip() for x in s.split(",") if x.strip()]



def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--global_model', default='data_juicer', type=str, help='ifle path to the LLaMA model')
    parser.add_argument('--data_path', default='./data', type=str,
                        help='file path to data')
    parser.add_argument('--cache_dir', default=None, type=str,
                        help='file path for caching data')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='output directory to store model and experiment result')
    parser.add_argument('--session_name', default='test', type=str,
                        help='name for your experiment')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='If set, save aggregated adapter_model.bin; otherwise only logs are kept')
    parser.add_argument('--deterministic', default=False, type=bool,
                        help='Enable deterministic CUDA kernels (slower, disables TF32/cuDNN benchmark)')
    parser.add_argument('--device_map', default='cuda', type=str,
                        help='HuggingFace device_map, e.g., "cuda", "auto", or "balanced"')
    parser.add_argument('--ablation', default=None, type=str,
                        choices=[None, 'uniform', 'random'],
                        help='FedHera ablation: None -> water-filling, uniform -> equal ranks, random -> random ranks within budgets')

    ## FL parameters
    parser.add_argument('--aggregation', default='homo', type=str,
                        help='aggregation method',
                        choices=['homo', 'flexlora', 'fedhera'])
    parser.add_argument('--hetero_mode', default='setting_B', type=str,
                        choices=['setting_A', 'setting_B'],
                        help='resource heterogeneity preset for Fed-Hera/FlexLoRA')
    parser.add_argument('--basis_update_every', default=1, type=int)
    parser.add_argument('--baseline', default='fedavg', type=str,
                        help='type of FL baseline to choose', choices=['fedavg', 'fedit'])
    parser.add_argument('--client_selection_frac', default=0.05, type=float,
                        help='ratio of how many clients participate in each round')
    parser.add_argument('--num_clients', default=1613, type=int,
                        help='total number of clients')
    parser.add_argument('--num_communication_rounds', default=50, type=int,
                        help='total number of communication rounds')
    parser.add_argument('--R_1', default=5, type=int,
                        help='Parameter for SLoRA. Total number of rounds for stage 1 sparse finetuning.')
    parser.add_argument('--early_stop', default=True, type=bool,
                        help='Early stop for FL training. If True, will apply early stop.')
    parser.add_argument('--patience', default=3, type=int,
                        help='Early stop patience.')
    parser.add_argument('--resume_epoch', default=None, type=int,
                        help='continue training from an existing experiment, specifying which comm round to resume')
    
    ## Local training parameters
    parser.add_argument('--local_batch_size', default=4, type=int,
                        help='local_batch_size')
    parser.add_argument('--local_micro_batch_size', default=2, type=int,
                        help='local_micro_batch_size')
    parser.add_argument('--dataloader_num_workers', default=4, type=int,
                        help='Number of worker processes for data loading')
    parser.add_argument('--local_num_epochs', default=1, type=int,
                        help='local epochs for local client training')
    parser.add_argument('--local_learning_rate', default=1e-6, type=float,
                        help='local training rate for local client training')
    parser.add_argument('--cutoff_len', default=512, type=int,
                        help='cut off len for tokenizing text')
    parser.add_argument('--warmup', default=0, type=int,
                        help='warm up steps for local training')
    parser.add_argument('--lr_decay', default=True, type=bool,
                        help='Learning rate decay. If true, will divide learning rate by 2 after 15-th comm round')
    parser.add_argument('--train_on_inputs', default=True, type=bool,
                        help='Whether training on input text')
    parser.add_argument('--group_by_length', default=False, type=bool,
                        help='')
    parser.add_argument('--prompt_template_name', default='alpaca', type=str,
                        help='template to generate prompt')

    ## LoRA Parameters
    parser.add_argument('--lora_r', default=8, type=int,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', default=16, type=int,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', default=0.05, type=float,
                        help='LoRA dropout')
    parser.add_argument('--lora_target_modules',
                        default=['q_proj', 'k_proj', 'v_proj'],
                        type=parse_lora_target_modules,
                        help='lora_target_modules (JSON list or comma-separated)',
                        )

    args = parser.parse_args()
    if isinstance(args.ablation, str) and args.ablation.lower() == "none":
        args.ablation = None
    return args


def model_and_tokenizer(global_model, device_map='cuda'):
    # model = AutoModelForCausalLM.from_pretrained(
    #     global_model,
    #     torch_dtype=torch.bfloat16,
    #     device_map=device_map,
    #     trust_remote_code=True,
    # )
    if CONFIG_MAPPING is not None and LlamaConfig is not None:
        model_id_lower = str(global_model).lower()
        if "mistral" in model_id_lower and "mistral" not in CONFIG_MAPPING:
            try:
                if hasattr(CONFIG_MAPPING, "register"):
                    CONFIG_MAPPING.register("mistral", LlamaConfig)
                else:
                    CONFIG_MAPPING._extra_content["mistral"] = LlamaConfig
            except Exception:
                pass
    # Accept friendly strings to force a single-GPU placement.
    map_arg = device_map
    if isinstance(device_map, str) and device_map.lower() in ["cuda", "gpu", "single", "0"]:
        map_arg = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        device_map=map_arg,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # For some newer models (e.g., Mistral) and older `tokenizers` versions,
    # the fast tokenizer JSON can be incompatible. Force the slow tokenizer
    # to avoid Rust `tokenizers` version issues.
    tokenizer = AutoTokenizer.from_pretrained(
        global_model,
        trust_remote_code=True,
        use_fast=False,
    )
    model_type = getattr(model.config, "model_type", "").lower()
    if tokenizer.pad_token_id is None:
        # For GPT-2-style models, padding with EOS is more stable.
        if model_type in ["gpt2"]:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
            else:
                tokenizer.pad_token_id = 0
        else:
            tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return model, tokenizer


def resolve_lora_targets_and_config_types(model, user_target_modules=None):
    """
    Choose sensible default LoRA target modules and heterogeneity configs
    based on the underlying model architecture.
    """
    model_type = getattr(model.config, "model_type", "").lower()

    # If the user explicitly provided a list, always respect it verbatim.
    if user_target_modules:
        target_modules = user_target_modules
    else:
        if model_type in ["llama", "mistral", "gemma"]:
            # LLaMA/Mistral/Gemma use the same projection names.
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ]
        elif model_type in ["gpt2"]:
            # GPT-2 blocks: attn.c_attn / attn.c_proj / mlp.c_fc / mlp.c_proj
            target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            # Fallback to the original default.
            target_modules = ['q_proj', 'v_proj']

    # Heterogeneous PEFT type presets.
    # Tie the three tiers directly to the canonical
    # resource levels so that FlexLoRA always picks
    # ranks from {1, 4, 16}.
    small_r = RESOURCE_RANKS["poor"]
    medium_r = RESOURCE_RANKS["medium"]
    large_r = RESOURCE_RANKS["high"]

    if model_type in ["llama", "mistral", "gemma"] and set(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    ).issubset(set(target_modules)):
        config_types = {
            'Type_0': {
                'q_proj': small_r, 'v_proj': small_r, 'k_proj': small_r, 'o_proj': small_r,
                'gate_proj': small_r, 'down_proj': small_r, 'up_proj': small_r,
            },
            'Type_1': {
                'q_proj': large_r, 'v_proj': large_r, 'k_proj': large_r, 'o_proj': large_r,
                'gate_proj': large_r, 'down_proj': large_r, 'up_proj': large_r,
            },
            'Type_2': {
                'q_proj': medium_r, 'v_proj': medium_r, 'k_proj': medium_r, 'o_proj': medium_r,
                'gate_proj': medium_r, 'down_proj': medium_r, 'up_proj': medium_r,
            },
            'Type_3': {
                'q_proj': medium_r, 'v_proj': medium_r, 'k_proj': medium_r, 'o_proj': medium_r,
                'gate_proj': medium_r, 'down_proj': medium_r, 'up_proj': medium_r,
            },
        }
    else:
        # Generic patterns for other architectures (including GPT-2).
        config_types = {
            'Type_0': {m: small_r for m in target_modules},
            'Type_1': {m: large_r for m in target_modules},
            'Type_2': {m: medium_r for m in target_modules},
            'Type_3': {m: medium_r for m in target_modules},
        }

    return target_modules, config_types


def _resource_probabilities(mode: str):
    """
    Map hetero mode to (low, medium, high) probabilities for FlexLoRA-style rank sampling.
    """
    if mode == 'setting_A':
        return [0.1, 0.3, 0.6]  # mostly high/medium ranks
    if mode == 'setting_B':
        return [0.3, 0.5, 0.2]  # 30% low, 50% mid, 20% high
    return [1/3, 1/3, 1/3]



def get_peft(config_types, num_clients, strategy=None, hetero_mode="setting_B", seed=42, fixed_ranks=None):
    """
    Get each client's unique LoRA configuration based on the aggregation strategy.
    """
    if strategy in ['homo', 'fedhera']:
        return {'alpha': 16, 'lora_dropout': 0.05}
    module_template = next(iter(config_types.values()), {})
    base_modules = list(module_template.keys())
    if fixed_ranks is not None:
        config_local = {'alpha': 16, 'lora_dropout': 0.05}
        for i in range(num_clients):
            rank = int(fixed_ranks.get(i, RESOURCE_RANKS["medium"]))
            config_local['Client_' + str(i)] = {m: rank for m in base_modules}
        return config_local
    rng = np.random.default_rng(seed)
    probs = _resource_probabilities(hetero_mode or "setting_B")
    tier_to_type = {"low": "Type_0", "medium": "Type_2", "high": "Type_1"}
    tiers = ["low", "medium", "high"]

    config_local = {'alpha': 16, 'lora_dropout': 0.05}
    for i in range(num_clients):
        tier = rng.choice(tiers, p=probs)
        type_key = tier_to_type[tier]
        config_local['Client_' + str(i)] = config_types[type_key]
    return config_local


def local_client_load_weight(args, model, epoch, global_params=None):
    """
    Load local client weight for non-FedHera strategies.
    """
    if args.aggregation == 'homo':
        _ = model.load_state_dict(global_params, strict=False)
    else:
        local_weight = load_weight_local(global_params, model)
        _ = model.load_state_dict(local_weight, strict=False)




def local_client_modify_layer(args, epoch, config_local, model, client_id):
    """
    Modify local client's LoRA layers based on local config.
    """
    if args.aggregation == 'fedhera':
        return
    if args.aggregation != 'homo':
        local_lora_config = config_local['Client_' + str(client_id)]
        modify_adapter(model, 'local', modify_module_rank=local_lora_config,
                       lora_alpha=config_local['alpha'], lora_dropout=config_local['lora_dropout'],
                       init_lora_weights=True)


def resume(args, data_path, output_dir, config_local):
    """
    Resume experiment from an existing study.
    """
    selected_clients_set = client_selection(args.num_clients, args.client_selection_frac,
                                            seed=args.seed, other_info=args.resume_epoch-1)
    local_dataset_len_dict = {}
    for client_id in tqdm(selected_clients_set):
        train_path = data_path + '/local_training_' + str(client_id) + '.json'
        train_data = load_dataset("json", data_files=train_path, cache_dir=args.cache_dir)
        local_dataset_len_dict[client_id] = len(train_data['train'])
    if args.aggregation == 'homo':
        global_params = FedAvg(selected_clients_set, output_dir, local_dataset_len_dict, args.resume_epoch-1)
    elif args.aggregation == 'flexlora':
        global_params = FlexLoRA(selected_clients_set, output_dir, local_dataset_len_dict, args.resume_epoch-1)
        global_params = distribute_weight_fast(global_params, config_local)
    else:
        global_params = None
    return global_params


def get_density(args, config_local, client_id, config_types):
    """
    get sparsity for slora sparse finetuning stage 1
    """
    if args.aggregation == 'homo':
        density = 0.0012
    else:
        if config_local['Client_' + str(client_id)] == config_types['Type_0']:
            density = 0.0012
        if config_local['Client_' + str(client_id)] == config_types['Type_1']:
            density = 0.1222
        if config_local['Client_' + str(client_id)] == config_types['Type_2']:
            density = 0.0822
        if config_local['Client_' + str(client_id)] == config_types['Type_3']:
            density = 0.0246
    return density


# training for FL setting
def FL_training(model, tokenizer, prompter, data_path, output_dir, args, config_local, config=None, config_types=None):
    logging.info("The process of federated instruction-tuning has started..")
    reset_traffic_stats()
    previously_selected_clients_set = set()
    output_dir = os.path.join(output_dir, str(args.num_clients))

    local_dataset_len_dict = dict()
    best_rouge_L = 0
    patience = args.patience
    current_count = 0
    if args.resume_epoch:
        global_params = resume(args, data_path, output_dir, config_local)
        start_epoch = args.resume_epoch
    else:
        start_epoch = 0
        global_params = None

    optim = 'sgd' if args.baseline == 'fedavg' else 'adamw_torch'
    for epoch in tqdm(range(start_epoch, args.num_communication_rounds)):
        local_train_results = 0
        local_eval_results = 0
        local_eval_rouge_1 = 0
        local_eval_rouge_L = 0
        total_data_num = 0
        logging.info("\In Epoch " + str(epoch))
        logging.info("\nConducting the client selection")

        selected_clients_set = client_selection(args.num_clients, args.client_selection_frac,
                                                seed=args.seed, other_info=epoch)
        if epoch == 15 and args.lr_decay:
            args.local_learning_rate = args.local_learning_rate / 2

        for k, client_id in enumerate(selected_clients_set):
            train_path = data_path + '/local_training_' + str(client_id) + '.json'
            train_data = load_dataset("json", data_files=train_path, cache_dir=args.cache_dir)
            local_dataset_len_dict[client_id] = len(train_data['train'])
            del train_data
            total_data_num += local_dataset_len_dict[client_id]

            local_client_modify_layer(args, epoch, config_local, model, client_id)

            from fed_utils.adaptive_peft import load_weight_fedhera_if_exists, apply_lora_prefix_mask
            prev_epoch = max(0, epoch - 1)
            pkg, meta = load_weight_fedhera_if_exists(output_dir, client_id, prev_epoch)
            hera_hooks = None
            if pkg is not None and meta is not None:
                per_layer_r_tot = {}
                for base_key, info in meta.items():
                    if info.get("skip", False):
                        continue
                    rt = int(info.get("r_tot", 0))
                    if rt <= 0:
                        continue
                    module_key = base_key.rsplit(".", 1)[0]
                    per_layer_r_tot[module_key] = rt

                if per_layer_r_tot:
                    modify_adapter(
                        model,
                        'local',
                        modify_module_rank=per_layer_r_tot,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        init_lora_weights=False,
                    )

                _ = model.load_state_dict(pkg, strict=False)

                per_layer_r_main = {
                    k: int(v.get("r_main", 0))
                    for k, v in meta.items()
                    if not v.get("skip", False)
                }
                hera_hooks = apply_lora_prefix_mask(model, per_layer_r_main)

            if epoch > 0 and args.aggregation != 'fedhera':
                local_client_load_weight(args, model, epoch, global_params=global_params)

            client = GeneralClient(client_id, model, tokenizer, prompter, data_path, output_dir, cache_dir=args.cache_dir,
                                   hetero_lora=False, optim=optim, dataloader_num_workers=args.dataloader_num_workers)

            logging.info("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset()

            local_eval_result = client.test(epoch, args.local_micro_batch_size)
            local_eval_results += float(local_eval_result['eval_loss']) * local_dataset_len_dict[client_id]
            local_eval_rouge_1 += float(local_eval_result['eval_rouge1']) * local_dataset_len_dict[client_id]
            local_eval_rouge_L += float(local_eval_result['eval_rougeL']) * local_dataset_len_dict[client_id]

            logging.info("Initiating the local training of Client_{}".format(client_id))

            client.build_local_trainer(tokenizer,
                                       args.local_micro_batch_size,
                                       args.local_batch_size // args.local_micro_batch_size,
                                       args.local_num_epochs,
                                       args.local_learning_rate,
                                       args.group_by_length,
                                       args.warmup)
            client.initiate_local_training()

            logging.info("Local training starts ... ")
            local_train_result = client.train()
            local_train_results += float(local_train_result['eval_loss']) * local_dataset_len_dict[client_id]

            logging.info("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            if 'hera_hooks' in locals() and hera_hooks is not None:
                for _h in hera_hooks:
                    try:
                        _h.remove()
                    except Exception:
                        pass
            del client

            logging.info("Collecting the weights of clients and performing aggregation")
        if args.aggregation == 'homo':
            global_params = FedAvg(selected_clients_set,
                                   output_dir,
                                   local_dataset_len_dict,
                                   epoch,
                                   )
            if args.save_model:
                torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
        elif args.aggregation == 'fedhera':
            FedHera(
                selected_clients_set,
                output_dir,
                local_dataset_len_dict,
                epoch,
                client_budgets=FL_training.client_budgets,
                layer_specs=FL_training.layer_specs,
                fixed_client_ranks=None,
                quant_scheme=("bfloat16", "nf4"),
                use_gpu_svd=True,
                basis_update_every=args.basis_update_every,
                ablation=args.ablation,
                lora_alpha=args.lora_alpha,
            )
            # adapter_model.bin 可存聚合Wg，便于可视化/对照
            # torch.save(_, os.path.join(output_dir, "adapter_model.bin"))
        elif args.aggregation == 'flexlora':
            global_params = FlexLoRA(selected_clients_set,
                                   output_dir,
                                   local_dataset_len_dict,
                                   epoch,
                                   )
            if args.save_model:
                torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
            global_params = distribute_weight_fast(global_params, config_local)
        else:
            raise ValueError(f"Unsupported aggregation mode: {args.aggregation}")

        global_eval_rouge_L = local_eval_rouge_L / total_data_num

        ### early stop
        if args.early_stop:
            if best_rouge_L < global_eval_rouge_L:
                best_rouge_L = global_eval_rouge_L
                best_round = epoch
                current_count = 0
            else:
                current_count += 1
            if current_count > patience:
                logging.info(f"Best round is {best_round} with test_rouge_L {best_rouge_L}")
                # Log final communication / compute statistics before exiting.
                try:
                    stats = get_traffic_stats()
                    logging.info("[TrafficSummary] %s", stats)
                except Exception:
                    pass
                return
        local_dataset_len_dict = {}
        import gc
        gc.collect()

    # Training finished without early stopping: record final traffic stats.
    try:
        stats = get_traffic_stats()
        logging.info("[TrafficSummary] %s", stats)
    except Exception:
        pass


def main():
    args = read_options()
    seed_torch(args.seed, deterministic=args.deterministic)
    if not os.path.exists(args.session_name):
        os.makedirs(args.session_name)
    log_dir = "/root/nfs/fedhera"
    os.makedirs(log_dir, exist_ok=True)
    model_tag = os.path.basename(str(args.global_model)).replace("/", "_")
    dataset_tag = os.path.basename(os.path.normpath(args.data_path))
    log_name = f"{args.aggregation}_{args.hetero_mode}_{model_tag}_{dataset_tag}_r{args.lora_r}.log"
    log_path = os.path.join(log_dir, log_name)
    logging.basicConfig(filename=log_path,
                        level=logging.INFO,
                        format='%(message)s')
    logging.info("Logging to %s", log_path)
    logging.info("Initial training parameters %s", args)
    print(args)


    logging.info(str(socket.gethostbyname(socket.gethostname())))

    data_path = os.path.join(args.data_path, str(args.num_clients))
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, args.session_name, args.aggregation)
    else:
        output_dir = os.path.join(args.session_name, args.aggregation)

    # set up the global model & tokenizer
    model, tokenizer = model_and_tokenizer(global_model=args.global_model, device_map=args.device_map)

    prompter = Prompter(args.prompt_template_name)

    # Build layer specs and budgets once for all aggregation strategies.
    layer_specs = extract_lora_layer_specs(model)
    client_budgets = get_client_budgets(args.num_clients, args.hetero_mode, seed=args.seed)
    calculated_ranks = calculate_unified_rank_from_budget(client_budgets, layer_specs)
    FL_training.layer_specs = layer_specs
    FL_training.client_budgets = client_budgets

    if args.aggregation == 'fedhera':
        fixed_ranks = None
    elif args.aggregation == 'flexlora':
        logging.info("Calculating FlexLoRA ranks based on client budgets...")
        fixed_ranks = calculated_ranks
    elif args.aggregation == 'homo':
        min_rank = min(calculated_ranks.values()) if calculated_ranks else 1
        logging.info(f"Homo: Bottleneck detected. Setting unified rank to {min_rank} for all clients.")
        fixed_ranks = {i: min_rank for i in range(args.num_clients)}
    else:
        raise ValueError(f"Unsupported aggregation method: {args.aggregation}")
    # Choose model-appropriate LoRA target modules and heterogeneity configs.
    lora_target_modules, config_types = resolve_lora_targets_and_config_types(
        model,
        user_target_modules=args.lora_target_modules,
    )
    config_local = get_peft(
        config_types,
        num_clients=args.num_clients,
        strategy=args.aggregation,
        hetero_mode=args.hetero_mode,
        seed=args.seed,
        fixed_ranks=fixed_ranks,
    )

    logging.info(config_local)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if args.baseline != 'slora':
        model = get_peft_model(model, config, adapter_name = 'local')
    
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    FL_training(model, tokenizer, prompter, data_path, output_dir, args, config_local=config_local, config=config, config_types=config_types)


if __name__ == "__main__":
    main()
