import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from peft import (
    get_peft_model_state_dict,
)
from .adaptive_peft import tokenize
import logging
import evaluate
import numpy as np


class GeneralClient:
    def __init__(self, client_id, model, tokenizer, prompter, data_path, output_dir, cutoff_len=512, train_on_inputs=True,
                 cache_dir=None, hetero_lora=False, optim='adamw_torch', dataloader_num_workers=4,
                 active_lora_layers=None):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.eval_data_path = os.path.join(data_path, "local_eval_{}.json".format(self.client_id))
        self.test_data_path = os.path.join(data_path, "local_test_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path, cache_dir=cache_dir)
        self.eval_data = load_dataset("json", data_files=self.eval_data_path, cache_dir=cache_dir)
        self.test_data = load_dataset("json", data_files=self.test_data_path, cache_dir=cache_dir)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.train_on_inputs = train_on_inputs
        self.cutoff_len = cutoff_len
        self.hetero_lora = hetero_lora
        self.optim = optim
        self.dataloader_num_workers = dataloader_num_workers
        self.pin_memory = torch.cuda.is_available()
        self.active_lora_layers = None if active_lora_layers is None else set(active_lora_layers)

    def compute_oracle_drift(self, global_params, oracle_r=4096, lora_alpha=16):
        """
        计算当前训练好的 Low-Rank 更新与 High-Rank Oracle 更新之间的差距。
        Reference: || Delta_W_method - Delta_W_oracle ||_F / || Delta_W_oracle ||_F
        """
        logging.info(f"Client {self.client_id}: Computing Oracle Drift (r={oracle_r})...")
        
        # 1. 缓存当前算法训练出的权重 (Method Weights)
        # 我们需要将其转换为 Dense Delta: (BA * scale)
        method_deltas = {}
        target_modules = set() # 记录哪些层被训练了
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "lora_A" in name and "default" not in name: # 假设当前 adapter 是 active 的
                    # name e.g., base_model.model.layers.0.self_attn.q_proj.lora_A.local.weight
                    base_key = ".".join(name.split(".")[:-3]) + ".lora"
                    target_modules.add(base_key)
                    
                    # 找到对应的 B
                    key_B = name.replace("lora_A", "lora_B")
                    param_B = self.model.get_parameter(key_B)
                    
                    # 计算 Low-Rank Delta
                    # W_delta = B @ A * (alpha / r)
                    r = param.shape[0]
                    scale = lora_alpha / r
                    
                    # shape: B=[d_out, r], A=[r, d_in] -> [d_out, d_in]
                    delta = (param_B @ param) * scale
                    method_deltas[base_key] = delta.detach().cpu()  # 移至 CPU 节省显存

        # 2. 回滚模型到初始状态 (Global State)
        # 注意：我们需要保存当前的 state_dict 以便最后恢复
        current_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        # 加载初始全局参数 (这会重置 LoRA 参数为本轮初始状态，或者对于 Oracle 来说，它是基座)
        # 实际上，我们需要一个新的干净的 Adapter。
        # 最简单的方法：Unload 当前 Adapter -> 加载新 Adapter -> Train -> Unload -> Reload 旧 Adapter
        
        # 3. 切换到 Oracle Adapter
        from peft import LoraConfig, get_peft_model
        
        # 卸载当前 adapter (逻辑上屏蔽即可，PEFT 支持多 adapter)
        adapter_name = "oracle_adapter"
        
        # 提取 target_modules 的简写名 (e.g. q_proj, v_proj) 用于 Config
        # 这是一个简化处理，假设所有层结构一致
        target_module_names = list(set([k.split(".")[-2] for k in method_deltas.keys()]))
        
        oracle_config = LoraConfig(
            r=oracle_r,
            lora_alpha=lora_alpha, # Alpha 通常设为 r 或 16，保持一致性即可，这里建议 alpha=r 以便 scale=1，或者保持 16
            target_modules=target_module_names,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 添加新的 Adapter
        self.model.add_adapter(adapter_name, oracle_config)
        self.model.set_adapter(adapter_name)
        
        # 确保 Oracle 是可训练的
        for n, p in self.model.named_parameters():
            if adapter_name in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
                
        # 4. 训练 Oracle
        # 重建 Trainer，因为参数变了
        # 为了速度，Oracle 可以训练较少的 step，但为了公平，应该保持 epoch 一致
        # 这里复用 build_local_trainer 的参数，但需要重新初始化 optimizer
        self.build_local_trainer(
            self.tokenizer,
            self.train_args.per_device_train_batch_size,
            self.train_args.gradient_accumulation_steps,
            self.train_args.num_train_epochs, # 保持 epoch 一致
            self.train_args.learning_rate,
            self.train_args.group_by_length,
            self.train_args.warmup_steps
        )
        
        # 开始训练 Oracle
        self.local_trainer.train()
        
        # 5. 计算 Oracle Delta 并 计算距离
        total_drift = 0.0
        total_oracle_norm = 0.0
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if adapter_name in name and "lora_A" in name:
                    base_key = ".".join(name.split(".")[:-3]) + ".lora"
                    
                    if base_key not in method_deltas:
                        continue
                        
                    key_B = name.replace("lora_A", "lora_B")
                    param_B = self.model.get_parameter(key_B)
                    
                    r = param.shape[0]
                    scale = lora_alpha / r
                    
                    # Delta Oracle
                    delta_oracle = (param_B @ param) * scale
                    delta_oracle = delta_oracle.cpu()
                    
                    delta_method = method_deltas[base_key]
                    
                    # 计算 Frobenius Norm
                    # Diff
                    diff = torch.norm(delta_method - delta_oracle, p='fro') ** 2
                    # Oracle Norm
                    oracle_norm = torch.norm(delta_oracle, p='fro') ** 2
                    
                    total_drift += diff.item()
                    total_oracle_norm += oracle_norm.item()
        
        # 最终指标
        import math
        relative_drift = math.sqrt(total_drift) / (math.sqrt(total_oracle_norm) + 1e-9)
        
        # 6. 清理现场
        # 删除 Oracle adapter
        self.model.delete_adapter(adapter_name)
        
        # 恢复原来的 Weights (重新加载之前的 state_dict)
        # 注意：self.model.load_state_dict(current_state_dict) 可能会有 strict 问题
        # 更安全的方式是切回 'local' (或 'default') adapter 并把参数赋回去
        self.model.set_adapter("local") 
        # 将参数从 cpu 拷回 gpu
        # 实际上 PEFT 的 delete_adapter 应该已经切回去了，但为了保险，我们要恢复之前训练好的值
        keys = self.model.load_state_dict(current_state_dict, strict=False)
        
        # 释放内存
        del method_deltas
        del current_state_dict
        torch.cuda.empty_cache()
        
        logging.info(f"Client {self.client_id} Drift Result: {relative_drift:.4f}")
        return relative_drift
    
    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(self.tokenizer, full_prompt, cutoff_len=self.cutoff_len, add_eos_token=True)
        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenizer(user_prompt, truncation=True, max_length=self.cutoff_len,
                                                   padding=False, return_tensors=None)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            )
        return tokenized_full_prompt

    def preprare_local_dataset(self, local_val_set_size=0):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(self.generate_and_tokenize_prompt)
            self.local_eval_dataset = self.eval_data["train"].shuffle().map(self.generate_and_tokenize_prompt)
            self.local_test_dataset = self.test_data["train"].shuffle().map(self.generate_and_tokenize_prompt)
        self.local_val_set_size = len(self.local_eval_dataset)

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            warmup=0,
                            lambd=None,
                            reg=None):
        
        def compute_metrics(pred):
            # 获取 Logits 和 Labels
            logits = pred.predictions
            # 如果 logits 是 tuple (比如包含 past_key_values)，取第一个元素
            if isinstance(logits, tuple):
                logits = logits[0]

            labels_ids = pred.label_ids

            # Argmax 获取预测的 Token ID [Batch, Seq_Len]
            pred_ids = np.argmax(logits, axis=-1)

            # -----------------------------------------------------------
            # 【关键修复】Shift 操作：对齐预测和标签
            # Causal LM 中，位置 t 的 Logit 预测的是 t+1 的 Label
            # -----------------------------------------------------------
            shift_preds = pred_ids[:, :-1]  # 预测值截掉最后一位
            shift_labels = labels_ids[:, 1:]  # 标签值截掉第一位

            # 创建掩码：忽略 padding 和 label 为 -100 的部分
            # pad_token_id 通常也是 -100 (在 DataCollator 中处理过) 或者 tokenizer.pad_token_id
            mask = (shift_labels != -100)

            # 计算 Token-level Accuracy
            # 只有在 mask 为 True 的位置才计算是否相等
            matches = (shift_preds == shift_labels) & mask
            correct = matches.sum()
            total_valid_tokens = mask.sum()

            accuracy = correct / max(1, total_valid_tokens)

            return {
                'accuracy': round(float(accuracy), 4),
            }

        use_cuda = torch.cuda.is_available()
        major, _ = torch.cuda.get_device_capability(0) if use_cuda else (0, 0)
        use_bf16 = use_cuda and major >= 8

        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            do_train=True,
            do_eval=True,
            fp16=use_cuda and not use_bf16,
            bf16=use_bf16,
            logging_steps=1,
            optim=self.optim,
            eval_strategy="epoch",
            save_strategy="no",
            output_dir=self.local_output_dir,
            group_by_length=group_by_length,
            dataloader_drop_last=False,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=use_cuda,
            gradient_checkpointing=True, 
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        active_set = self.active_lora_layers
        for name, p in self.model.named_parameters():
            if 'lora_' not in name:
                p.requires_grad = False
            else:
                if active_set is None:
                    p.requires_grad = True
                else:
                    base_key = '.'.join(name.split('.')[:-3]) + '.lora'
                    p.requires_grad = base_key in active_set
        lora_params = [p for n, p in self.model.named_parameters() if ('lora_' in n and p.requires_grad)]
        if len(lora_params) == 0:
            raise ValueError("No LoRA parameters found to optimize. Ensure adapters are added via get_peft_model.")

        try:
            from bitsandbytes.optim import Adam8bit
            optimizer = Adam8bit(lora_params, lr=local_learning_rate, weight_decay=0.0)
        except Exception:
            optimizer = torch.optim.AdamW(lora_params, lr=local_learning_rate, weight_decay=0.0)

        steps_per_epoch = max(1, len(self.local_train_dataset) // max(1, local_micro_batch_size))
        update_steps_per_epoch = max(1, steps_per_epoch // max(1, gradient_accumulation_steps))
        total_steps = max(1, update_steps_per_epoch * max(1, int(local_num_epochs)))
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )

        self.local_trainer = transformers.Trainer(model=self.model,
                                                  train_dataset=self.local_train_dataset,
                                                  eval_dataset=self.local_eval_dataset,
                                                  args=self.train_args,
                                                  data_collator=transformers.DataCollatorForSeq2Seq(
                                                      tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                      padding=True
                                                  ),
                                                  optimizers=(optimizer, scheduler),
                                                  compute_metrics=compute_metrics
                                                  )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if "lora" in name))
        self.params_dict_new = OrderedDict(
            (name, param.detach()) for name, param in self.model.named_parameters() if "lora" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "lora"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        result = self.local_trainer.train()
        logging.info(self.local_trainer.state.log_history[-2])
        logging.info(self.local_trainer.state.log_history[-1])
        logging.info(result.metrics)
        return self.local_trainer.state.log_history[-2]

    def test(self, epoch, local_micro_batch_size):
        use_cuda = torch.cuda.is_available()
        major, _ = torch.cuda.get_device_capability(0) if use_cuda else (0, 0)
        use_bf16 = use_cuda and major >= 8

        test_args = transformers.TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_eval=True,
            fp16=use_cuda and not use_bf16,
            bf16=use_bf16,
            per_device_eval_batch_size=local_micro_batch_size,
            dataloader_drop_last=False,
            eval_accumulation_steps=4,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=use_cuda,
        )
        rouge_metric = evaluate.load("./evaluate/metrics/rouge/rouge.py")
        bleu_metric = evaluate.load("./evaluate/metrics/bleu")
        meteor_metric = evaluate.load("./evaluate/metrics/meteor")
        nist_metric = evaluate.load("./evaluate/metrics/nist_mt")
        try:
            from pycocoevalcap.cider.cider import Cider
            cider_scorer = Cider()
        except Exception:
            cider_scorer = None
            logging.warning("CIDEr unavailable (install pycocoevalcap).")
        
        def compute_metrics(pred):
            # ---------------------------------------------------------------------
            # 1. 预处理：获取 Logits 和 Labels
            # ---------------------------------------------------------------------
            logits = pred.predictions
            # 如果 logits 是 tuple (比如包含 past_key_values)，取第一个元素
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # 获取原始的 ID
            pred_ids = np.argmax(logits, axis=-1)
            labels_ids = pred.label_ids # 这里的 labels 包含 -100

            # ---------------------------------------------------------------------
            # 2. 关键修复：Shift 操作 (对齐预测和标签)
            # ---------------------------------------------------------------------
            # Causal LM 中，位置 t 的 Logit 预测的是 t+1 的 Label
            # 所以我们需要把 预测值左移 (去掉最后一位) 和 标签值右移 (去掉第一位) 来对齐
            shift_preds = pred_ids[:, :-1]
            shift_labels = labels_ids[:, 1:]

            # ---------------------------------------------------------------------
            # 3. 计算 Token-level Accuracy (替换了原先严苛的 String Accuracy)
            # ---------------------------------------------------------------------
            # 创建掩码：忽略 padding 和 label 为 -100 的部分
            mask = (shift_labels != -100)

            # 只有在 mask 为 True 的位置才计算是否相等
            matches = (shift_preds == shift_labels) & mask
            correct = matches.sum()
            total_valid_tokens = max(1, mask.sum())

            token_accuracy = correct / total_valid_tokens

            # ---------------------------------------------------------------------
            # 4. 保留 ROUGE 计算 (优化：使用移位后的数据解码，确保文本对齐)
            # ---------------------------------------------------------------------
            # 准备解码用的 Pad ID
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            # 处理标签中的 -100，将其替换为 pad_id 以便 tokenizer 解码
            # 注意：这里我们使用 shift_labels，这样解码出来的文本和预测文本在语义上是对齐的
            clean_labels = np.where(shift_labels == -100, pad_id, shift_labels)

            # 解码为字符串
            pred_str = self.tokenizer.batch_decode(shift_preds, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(clean_labels, skip_special_tokens=True)

            rouge_out = rouge_metric.compute(predictions=pred_str, references=label_str, use_aggregator=True)
            bleu_out = bleu_metric.compute(predictions=pred_str, references=label_str)
            meteor_out = meteor_metric.compute(predictions=pred_str, references=label_str)
            nist_out = nist_metric.compute(predictions=pred_str, references=label_str)
            cider_score = None
            if cider_scorer is not None:
                gts = {i: [ref] for i, ref in enumerate(label_str)}
                res = {i: [hyp] for i, hyp in enumerate(pred_str)}
                cider_score, _ = cider_scorer.compute_score(gts, res)

            # ---------------------------------------------------------------------
            # 5. 返回合并后的指标
            # ---------------------------------------------------------------------
            metrics = {
                "accuracy": round(float(token_accuracy), 4),
                "rouge1": round(rouge_out["rouge1"], 4),
                "rouge2": round(rouge_out["rouge2"], 4),
                "rougeL": round(rouge_out["rougeL"], 4),
                "rougeLsum": round(rouge_out["rougeLsum"], 4),
                "bleu": round(bleu_out["bleu"], 4),
                "meteor": round(meteor_out["meteor"], 4),
                "nist": round(nist_out["nist_mt"], 4),
            }
            if cider_score is not None:
                metrics["cider"] = round(float(cider_score), 4)
            return metrics

        tester = transformers.Trainer(
            model=self.model,
            args=test_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            compute_metrics=compute_metrics
        )
        eval_dataset = self.local_eval_dataset
        eval_results = tester.evaluate(eval_dataset)
        logging.info('For client ' + str(self.client_id) + ', the eval result is:')
        logging.info(eval_results)
        return eval_results

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        lora_params = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name and param.requires_grad:
                lora_params[name] = param
        single_output_dir = os.path.join(self.output_dir, str(self.client_id), "local_output_epoch_{}".format(epoch))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(lora_params, single_output_dir + "/pytorch_model.bin")

        _ = self.model.load_state_dict(self.params_dict_old, strict=False)
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id
        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id
