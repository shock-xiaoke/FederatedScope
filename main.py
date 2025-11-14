from tqdm import tqdm
from scipy.stats import norm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from fed_utils import FedAvg, client_selection, seed_torch, GeneralClient, FlexLoRA, \
    load_weight_local, distribute_weight_fast, modify_adapter, load_weight_SLoRA, FedHera

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

    ## FL parameters
    # parser.add_argument('--aggregation', default='homo', type=str,
    #                     help='aggregation method', choices=['homo','random','heavy_tail','heavy_tail_strong','normal'])
    parser.add_argument('--aggregation', default='homo', type=str, help = 'aggregation method',
                        choices = ['homo', 'random', 'heavy_tail', 'heavy_tail_strong', 'normal', 'fedhera'])
    parser.add_argument('--hetero_mode', default='heavy_tail', type=str,
                        choices = ['random', 'normal', 'heavy_tail'], help = 'resource heterogeneity mode for Fed-Hera')
    parser.add_argument('--basis_update_every', default=5, type=int)
    parser.add_argument('--baseline', default='fedavg', type=str,
                        help='type of FL baselines to choose', choices=['fedavg', 'slora', 'fedit'])
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
    # parser.add_argument('--lora_target_modules', default=['q_proj', 'v_proj', 'k_proj', 'o_proj',
    #                                                       'gate_proj', 'down_proj', 'up_proj'
    #                                                       ], type=list,
    #                     help='lora_target_modules')
    parser.add_argument('--lora_target_modules', default=['q_proj', 'v_proj'], type=list,
                        help='lora_target_modules')

    args = parser.parse_args()
    return args


def model_and_tokenizer(global_model, device_map='auto'):
    """
    Load model and tokenizer and place the model on GPU if available.
    Prefer bf16 on Ampere+ GPUs, otherwise fall back to fp16/cpu fp32.
    """
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        use_bf16 = major >= 8  # Ampere or newer
        if use_bf16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
        device = 'cuda'
    else:
        use_bf16 = False
        torch_dtype = torch.float32
        device = 'cpu'

    # Do not rely on Accelerate's device_map here; move explicitly.
    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(global_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return model, tokenizer


def get_peft(config_types, num_clients, strategy=None):
    """
    get each client's unique LoRA configuration based on the "aggregation" parameter
    """
    if strategy == 'homo':
        return
    else:
        # random select lora type for clients
        if strategy == 'random':
            config_local = {'alpha':16, 'lora_dropout':0.05}
            for i in range(num_clients):
                type = 'Type_' + str(np.random.randint(0, 4))
                config_local['Client_' + str(i)] = config_types[type]
        elif strategy == 'fedhera':
            config_local = {'alpha': 16, 'lora_dropout': 0.05}
            for i in range(num_clients):
                config_local[f'Client_{i}'] = {}
            return config_local
        elif strategy == 'heavy_tail':
            config_local = {'alpha':16, 'lora_dropout':0.05}
            for i in range(num_clients):
                rand_num = random.random()  # Generate a random float between 0 and 1
                if rand_num < 0.80:
                    type = 'Type_0'
                elif rand_num < 0.90:
                    type = 'Type_1'
                elif rand_num < 0.95:
                    type = 'Type_2'
                else:
                    type = 'Type_3'
                config_local['Client_' + str(i)] = config_types[type]
        elif strategy == 'heavy_tail_strong':
            config_local = {'alpha':16, 'lora_dropout':0.05}
            for i in range(num_clients):
                rand_num = random.random()  # Generate a random float between 0 and 1
                if rand_num < 0.80:
                    type = 'Type_1'
                elif rand_num < 0.90:
                    type = 'Type_2'
                elif rand_num < 0.95:
                    type = 'Type_3'
                else:
                    type = 'Type_0'
                config_local['Client_' + str(i)] = config_types[type]
        elif strategy == 'normal':
            config_local = {'alpha': 16, 'lora_dropout': 0.05}
            positions = np.array([0, 3, 2, 1])
            mu = 1.5
            sigma = 0.7
            probabilities = norm.pdf(positions, mu, sigma)
            probabilities /= probabilities.sum()
            for i in range(num_clients):
                selected_var = np.random.choice(positions, p=probabilities)
                type = 'Type_' + str(selected_var)
                config_local['Client_' + str(i)] = config_types[type]

        return config_local

def local_client_load_weight(args, model, epoch, global_params=None):
    """
    load local client's weight
    """
    if args.baseline == 'slora' and epoch != args.R_1:
        if epoch < args.R_1 or args.aggregation == 'homo':
            local_weight = global_params
            _ = model.load_state_dict(local_weight, strict=False)
        else:
            local_weight = load_weight_local(global_params, model)
            _ = model.load_state_dict(local_weight, strict=False)
    else:
        if args.aggregation == 'homo':
            _ = model.load_state_dict(global_params, strict=False)
        else:
            local_weight = load_weight_local(global_params, model)
            _ = model.load_state_dict(local_weight, strict=False)



def local_client_modify_layer(args, epoch, config_local, model, client_id):
    """
    Modify local client's LoRA layers based on local config
    """
    if args.aggregation != 'homo':
        if args.baseline == 'slora' and epoch >= args.R_1:
            local_lora_config = config_local['Client_' + str(client_id)]
            modify_adapter(model, 'local', modify_module_rank=local_lora_config,
                           lora_alpha=config_local['alpha'], lora_dropout=config_local['lora_dropout'],
                           init_lora_weights=True)
        else:
            local_lora_config = config_local['Client_' + str(client_id)]
            modify_adapter(model, 'local', modify_module_rank=local_lora_config,
                           lora_alpha=config_local['alpha'], lora_dropout=config_local['lora_dropout'],
                           init_lora_weights=True)

def resume(args, data_path, output_dir, config_local):
    """
    resume experiment from an existing study
    """
    selected_clients_set = client_selection(args.num_clients, args.client_selection_frac,
                                            seed=args.seed, other_info=args.resume_epoch-1)
    local_dataset_len_dict = []
    for client_id in tqdm(selected_clients_set):
        train_path = data_path + '/local_training_' + str(client_id) + '.json'
        train_data = load_dataset("json", data_files=train_path, cache_dir=args.cache_dir)
        local_dataset_len_dict[client_id] = len(train_data['train'])
    if args.baseline == 'slora':
        if args.resume_epoch-1 < args.R_1 or args.aggregation == 'fedavg':
            global_params = FedAvg(selected_clients_set,
                                   output_dir,
                                   local_dataset_len_dict,
                                   args.resume_epoch-1,
                                   )
        else:
            global_params = FlexLoRA(selected_clients_set,
                                     output_dir,
                                     local_dataset_len_dict,
                                     args.resume_epoch-1,
                                     )
            global_params = distribute_weight_fast(global_params, config_local)
    else:
        if args.aggregation == 'homo':
            global_params = FedAvg(selected_clients_set,
                                   output_dir,
                                   local_dataset_len_dict,
                                   args.resume_epoch-1,
                                   )
        else:
            global_params = FlexLoRA(selected_clients_set,
                                     output_dir,
                                     local_dataset_len_dict,
                                     args.resume_epoch-1,
                                     )
            global_params = distribute_weight_fast(global_params, config_local)
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

    optim = 'sgd' if args.baseline == 'fedavg' else 'adamw_torch'
    for epoch in tqdm(range(start_epoch, args.num_communication_rounds)):
        local_train_results = 0
        local_eval_results = 0
        local_eval_rouge_1 = 0
        local_eval_rouge_L = 0
        total_data_num = 0
        logging.info("\In Epoch " + str(epoch))
        logging.info("\nConducting the client selection")

        #select participating clients
        selected_clients_set = client_selection(args.num_clients, args.client_selection_frac,
                                                seed=args.seed, other_info=epoch)
        if epoch == 15 and args.lr_decay == True:
            args.local_learning_rate=args.local_learning_rate/2

        if args.baseline == 'slora' and epoch == args.R_1:
            model, tokenizer = model_and_tokenizer(global_model=args.global_model, device_map='auto')
            # IMPORTANT: get_peft_model returns a wrapped model; assign it back
            model = get_peft_model(model, config, adapter_name='local')

        # training for each client
        for k, client_id in enumerate(selected_clients_set):
            train_path = data_path + '/local_training_' + str(client_id) + '.json'
            train_data = load_dataset("json", data_files=train_path, cache_dir=args.cache_dir)
            local_dataset_len_dict[client_id] = len(train_data['train'])
            total_data_num += local_dataset_len_dict[client_id]

            # Fed-Hera: 尝试加载 server_push 包（若本轮生成）
            from fed_utils.adaptive_peft import load_weight_fedhera_if_exists, apply_lora_prefix_mask
            prev_epoch = max(0, epoch - 1)
            pkg, meta = load_weight_fedhera_if_exists(output_dir, client_id, prev_epoch)
            if pkg is not None:
                # Resize LoRA ranks per layer to r_tot before loading server weights
                per_layer_r_tot = {k: int(v.get("r_tot", 0)) for k, v in meta.items() if not v.get("skip", False)}
                if len(per_layer_r_tot) > 0:
                    modify_adapter(model, 'local', modify_module_rank=per_layer_r_tot,
                                   lora_alpha=16, lora_dropout=0.05, init_lora_weights=False)
                _ = model.load_state_dict(pkg, strict=False)
                # 根据 meta 设置每层 r_main 的前缀门控
                per_layer_r_main = {k: int(v.get("r_main", 0)) for k, v in meta.items() if not v.get("skip", False)}
                hera_hooks = apply_lora_prefix_mask(model, per_layer_r_main)

            if args.baseline == 'slora' and args.R_1 == epoch:
                local_weight = load_weight_SLoRA(global_params, model)
                _ = model.load_state_dict(local_weight, strict=False)

            if epoch > 0 and args.aggregation != 'fedhera':
                local_client_load_weight(args, model, epoch, global_params=global_params)

            client = GeneralClient(client_id, model, tokenizer, prompter, data_path, output_dir, cache_dir=args.cache_dir,
                                   hetero_lora = False, optim = optim)

            logging.info("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset()

            local_eval_result = client.test(epoch,args.local_micro_batch_size)
            local_eval_results += float(local_eval_result['eval_loss']) * local_dataset_len_dict[client_id]
            local_eval_rouge_1 += float(local_eval_result['eval_rouge1']) * local_dataset_len_dict[client_id]
            local_eval_rouge_L += float(local_eval_result['eval_rougeL']) * local_dataset_len_dict[client_id]

            logging.info("Initiating the local training of Client_{}".format(client_id))

            if args.baseline == 'slora' and epoch < args.R_1:
                density = get_density(args, config_local, client_id, config_types)
                sparse = True
                client.get_sparse(model, args.local_learning_rate, args.local_micro_batch_size, args.warmup, density)
            else:
                sparse = False

            client.build_local_trainer(tokenizer,
                                       args.local_micro_batch_size,
                                       args.local_batch_size // args.local_micro_batch_size,
                                       args.local_num_epochs,
                                       args.local_learning_rate,
                                       args.group_by_length,
                                       args.warmup)
            client.initiate_local_training(sparse)

            logging.info("Local training starts ... ")
            local_train_result = client.train()
            local_train_results += float(local_train_result['eval_loss']) * local_dataset_len_dict[client_id]

            logging.info("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            # Clean up Fed-Hera gradient hooks to avoid accumulation across clients
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
            torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
        elif args.aggregation == 'fedhera':
            # Fed-Hera: 生成每客户端下发包 + 返回全局Wg（可选保存做日志）
                      _ = FedHera(selected_clients_set,
                                                           output_dir,
                                                           local_dataset_len_dict,
                                                           epoch,
                                                           client_budgets = FL_training.client_budgets,
                                  layer_specs = FL_training.layer_specs,
                                  quant_scheme = ("bfloat16", "nf4"),
                                  use_gpu_svd = False,
                                  basis_update_every = args.basis_update_every)
            # adapter_model.bin 可存聚合Wg，便于可视化/对照
                      torch.save(_, os.path.join(output_dir, "adapter_model.bin"))
        else:
            global_params = FlexLoRA(selected_clients_set,
                                   output_dir,
                                   local_dataset_len_dict,
                                   epoch,
                                   )
            torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
            global_params = distribute_weight_fast(global_params, config_local)

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
                return

def main():
    args = read_options()
    seed_torch(args.seed)
    if not os.path.exists(args.session_name):
        os.makedirs(args.session_name)
    if args.output_dir:
        if not os.path.exists(os.path.join(args.output_dir, args.session_name)):
            os.makedirs(os.path.join(args.output_dir, args.session_name))
        logging.basicConfig(filename=os.path.join(args.output_dir, args.session_name, '../result.log'),
                            level=logging.INFO,
                            format='%(message)s')
    else:
        logging.basicConfig(filename=os.path.join(args.session_name, '../result.log'),
                            level=logging.INFO,
                            format='%(message)s')
    logging.info("Initial training parameters %s", args)
    print(args)


    logging.info(str(socket.gethostbyname(socket.gethostname())))

    data_path = os.path.join(args.data_path, str(args.num_clients))
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, args.session_name, args.aggregation)
    else:
        output_dir = os.path.join(args.session_name, args.aggregation)

    # set up the global model & toknizer
    model, tokenizer = model_and_tokenizer(global_model=args.global_model, device_map='auto')

    prompter = Prompter(args.prompt_template_name)

    # 对每层记录 (d_out, d_in)，提供给 Fed-Hera 的分配器做字节估算
    layer_specs = {}

    for name, param in model.named_parameters():

        if "lora_A" in name or "lora_B" in name:
            base_key = '.'.join(name.split('.')[:-3]) + '.lora'
            if base_key not in layer_specs:
                if "lora_A" in name:
                    r, d_in = param.shape
                    # 对应 B 的形状稍后由伙伴参数获取
                    layer_specs[base_key] = {"d_out": None, "d_in": int(d_in)}
                else:
                    d_out, r = param.shape
                    if base_key not in layer_specs:
                        layer_specs[base_key] = {"d_out": int(d_out), "d_in": None}
                    else:
                        layer_specs[base_key]["d_out"] = int(d_out)
    # 补全空值

    for k, v in layer_specs.items():
        if v["d_out"] is None or v["d_in"] is None:
            # 粗略补全：如果某层只出现一侧，取另一侧出现的维度
            for name, p in model.named_parameters():
                if k in name:
                    if v["d_out"] is None and "lora_B" in name:
                        v["d_out"] = int(p.shape[0])
                    if v["d_in"] is None and "lora_A" in name:
                        v["d_in"] = int(p.shape[1])

    # 生成 Fed-Hera 的 per-client 预算
    client_budgets = build_fedhera_budgets(args.num_clients, args.hetero_mode, seed=args.seed)
    # 传入训练循环（避免函数签名大改）
    FL_training.layer_specs = layer_specs
    FL_training.client_budgets = client_budgets

    config_types = {
        'Type_0': {'q_proj': 8, 'v_proj': 8, 'k_proj': 8, 'o_proj': 8, 'gate_proj': 8, 'down_proj': 8, 'up_proj': 8},
        'Type_1': {'q_proj': 200, 'v_proj': 200, 'k_proj': 200, 'o_proj': 200, 'gate_proj': 200, 'down_proj': 200,
                   'up_proj': 200},
        'Type_2': {'q_proj': 30, 'v_proj': 30, 'k_proj': 30, 'o_proj': 30, 'gate_proj': 200, 'down_proj': 200,
                   'up_proj': 200},
        'Type_3': {'q_proj': 30, 'v_proj': 30, 'k_proj': 30, 'o_proj': 30, 'gate_proj': 30, 'down_proj': 30,
                   'up_proj': 30}, }
    config_local = get_peft(config_types, num_clients=args.num_clients, strategy=args.aggregation)

    logging.info(config_local)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if args.baseline != 'slora':
        model = get_peft_model(model, config, adapter_name='local')

    # Rebuild layer_specs for Fed-Hera after LoRA modules are attached
    if args.aggregation == 'fedhera':
        layer_specs = {}
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                base_key = '.'.join(name.split('.')[:-3]) + '.lora'
                if base_key not in layer_specs:
                    if "lora_A" in name:
                        _, d_in = param.shape
                        layer_specs[base_key] = {"d_out": None, "d_in": int(d_in)}
                    else:
                        d_out, _ = param.shape
                        if base_key not in layer_specs:
                            layer_specs[base_key] = {"d_out": int(d_out), "d_in": None}
                        else:
                            layer_specs[base_key]["d_out"] = int(d_out)
        for k, v in layer_specs.items():
            if v["d_out"] is None or v["d_in"] is None:
                for name, p in model.named_parameters():
                    if k in name:
                        if v["d_out"] is None and "lora_B" in name:
                            v["d_out"] = int(p.shape[0])
                        if v["d_in"] is None and "lora_A" in name:
                            v["d_in"] = int(p.shape[1])
        FL_training.layer_specs = layer_specs

    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    FL_training(model, tokenizer, prompter, data_path, output_dir, args, config_local=config_local, config=config, config_types=config_types)


def _sample_factor(mode, rng):
    if mode == 'random':
        return rng.uniform(0.8, 1.2)
    elif mode == 'normal':
        f = rng.normal(loc=1.0, scale=0.1)
        return float(np.clip(f, 0.7, 1.3))
    else:  # heavy_tail
        f = rng.lognormal(mean=-0.1, sigma=0.5)
        return float(np.clip(f, 0.5, 2.0))

def build_fedhera_budgets(num_clients, hetero_mode, seed=42):
    """
    返回 Dict[int]-> {"tier": str, "B_down_MB":float, "VRAM_MB":float, "step_ms":float}
    low/medium/high 对应采样频率由 hetero_mode 决定：heavy_tail 偏向 low。
    """
    rng = np.random.default_rng(seed)
    # tier 基线
    TIERS = {
        "low":    {"B_down_MB": 80,  "VRAM_MB": 12000, "step_ms": 350},
        "medium": {"B_down_MB": 140, "VRAM_MB": 20000, "step_ms": 250},
        "high":   {"B_down_MB": 240, "VRAM_MB": 32000, "step_ms": 200},
    }
    # 采样概率
    if hetero_mode == 'random':
        probs = [1/3, 1/3, 1/3]
    elif hetero_mode == 'normal':
        probs = [0.25, 0.5, 0.25]
    else:  # heavy_tail -> 弱算力更多
        probs = [0.6, 0.3, 0.1]
    tier_names = ["low","medium","high"]

    client_budgets = {}
    for i in range(num_clients):
        tier = rng.choice(tier_names, p=probs)
        base = TIERS[tier]
        # 对每项资源加入扰动
        f_down = _sample_factor(hetero_mode, rng)
        f_vram = _sample_factor(hetero_mode, rng)
        f_step = _sample_factor(hetero_mode, rng)
        client_budgets[i] = {
            "tier": tier,
            "B_down_MB": float(base["B_down_MB"] * f_down),
            "VRAM_MB":   float(base["VRAM_MB"]   * f_vram),
            "step_ms":   float(base["step_ms"]   / max(f_step, 1e-6))  # 算力强→步时更小
        }
    return client_budgets


if __name__ == "__main__":
    main()
