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
                 cache_dir=None, hetero_lora=False, optim='adamw_torch', dataloader_num_workers=4):
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
        ddp = False

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
            evaluation_strategy="epoch",
            save_strategy="no",
            output_dir=self.local_output_dir,
            group_by_length=group_by_length,
            dataloader_drop_last=False,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=use_cuda,
        )

        for name, p in self.model.named_parameters():
            if 'lora_' not in name:
                p.requires_grad = False
        lora_params = [p for n, p in self.model.named_parameters() if ('lora_' in n and p.requires_grad)]
        if len(lora_params) == 0:
            raise ValueError("No LoRA parameters found to optimize. Ensure adapters are added via get_peft_model.")

        try:
            from bitsandbytes.optim import Adam8bit
            optimizer = Adam8bit(lora_params, lr=local_learning_rate)
        except Exception:
            optimizer = torch.optim.AdamW(lora_params, lr=local_learning_rate)

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

        def compute_metrics(pred):
            labels_ids = np.array(pred.label_ids)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            labels_ids = np.where(labels_ids == -100, pad_id, labels_ids)
            pred_ids = np.argmax(pred.predictions, axis=-1)
            pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

            def _norm(x): return x.strip().lower()

            correct = 0
            for p, l in zip(pred_str, label_str):
                p_n = _norm(p)
                l_n = _norm(l)
                if p_n == l_n or l_n in p_n:
                    correct += 1
            accuracy = correct / max(1, len(label_str))

            rouge = evaluate.load('./evaluate/metrics/rouge/rouge.py')
            rouge_output = rouge.compute(predictions=pred_str, references=label_str, use_aggregator=True)
            return {
                'rouge1': round(rouge_output["rouge1"], 4),
                'rouge2': round(rouge_output["rouge2"], 4),
                'rougeL': round(rouge_output["rougeL"], 4),
                'rougeLsum': round(rouge_output["rougeLsum"], 4),
                'accuracy': round(accuracy, 4),
            }

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
