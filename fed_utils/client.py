import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from peft import get_peft_model_state_dict
from .adaptive_peft import tokenize
import logging
import evaluate
import numpy as np
import re
from fractions import Fraction
from typing import Dict, List, Tuple, Optional
import inspect
import transformers

def make_training_arguments(**kwargs):
    """
    Build transformers.TrainingArguments in a version-robust way:
    - filter out unsupported kwargs
    - rename a few known changed argument names
    """
    sig = inspect.signature(transformers.TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())

    # rename mapping for different transformers versions
    rename_map = {
        "evaluation_strategy": "eval_strategy",   # some versions rename/deprecate
        "save_strategy": "save_strategy",         # keep, but here for symmetry
        "logging_strategy": "logging_strategy",
    }

    fixed = dict(kwargs)

    # rename if needed
    for old, new in rename_map.items():
        if old in fixed and old not in allowed and new in allowed:
            fixed[new] = fixed.pop(old)

    # filter unsupported
    filtered = {k: v for k, v in fixed.items() if k in allowed}

    return transformers.TrainingArguments(**filtered)


# -----------------------------
# Helpers: parsing & normalization
# -----------------------------
_LABEL_RE = re.compile(r"\b(A|B|C|D|TRUE|FALSE)\b", re.IGNORECASE)


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_mc_label(text: str) -> str:
    """Extract first occurrence of A/B/C/D/True/False (case-insensitive). Return normalized uppercase label."""
    t = _norm_space(text).upper()
    m = _LABEL_RE.search(t)
    if not m:
        return ""
    lab = m.group(1).upper()
    return lab


_NUM_RE = re.compile(r"(-?\d+/\d+|-?\d*\.\d+|-?\d+)")


def _normalize_math_token(x: str) -> str:
    """Normalize number/fraction token to a canonical string for EM matching."""
    x = _norm_space(x)
    if not x:
        return ""
    # Remove surrounding latex wrappers if any
    x = x.replace("\\boxed", "").replace("{", "").replace("}", "")
    x = x.replace("\\,", "").replace(",", "")
    x = x.strip()

    # Try fraction a/b
    if "/" in x:
        parts = x.split("/")
        if len(parts) == 2:
            try:
                frac = Fraction(int(parts[0].strip()), int(parts[1].strip()))
                return f"{frac.numerator}/{frac.denominator}"
            except Exception:
                pass

    # Try int
    try:
        iv = int(x)
        return str(iv)
    except Exception:
        pass

    # Try float (keep minimal string)
    try:
        fv = float(x)
        # Avoid scientific notation issues
        if abs(fv - round(fv)) < 1e-9:
            return str(int(round(fv)))
        # Trim trailing zeros
        s = f"{fv:.10f}".rstrip("0").rstrip(".")
        return s
    except Exception:
        return x


def extract_gsm8k_final(text: str) -> str:
    """Prefer '#### <ans>' else fallback to last number-like token."""
    t = text or ""
    m = re.findall(r"####\s*([^\n]+)", t)
    if m:
        return _normalize_math_token(m[-1])
    m2 = _NUM_RE.findall(t)
    if m2:
        return _normalize_math_token(m2[-1])
    return _normalize_math_token(t)


def extract_metamath_final(text: str) -> str:
    """MetaMathQA/arithmetic: fallback to last number-like token; also handle 'The answer is:' patterns."""
    t = text or ""
    m2 = _NUM_RE.findall(t)
    if m2:
        return _normalize_math_token(m2[-1])
    return _normalize_math_token(t)


def exact_match(a: str, b: str) -> float:
    return 1.0 if _norm_space(a) == _norm_space(b) else 0.0


# -----------------------------
# Client
# -----------------------------
class GeneralClient:
    def __init__(
        self,
        client_id,
        model,
        tokenizer,
        prompter,
        data_path,
        output_dir,
        cutoff_len=512,
        train_on_inputs=True,
        cache_dir=None,
        hetero_lora=False,
        optim='adamw_torch',
        dataloader_num_workers=4,
        # NEW eval knobs
        eval_protocol: str = "auto",          # auto|gen|legacy_tf
        eval_answer_only_loss: bool = True,   # mask prompt for eval/test loss
        eval_max_samples: int = 0,            # 0 means all
        eval_gen_batch_size: int = 4,
        eval_gen_max_new_tokens: int = 128,   # default for NLG; per-task override inside
    ):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.local_data_path = os.path.join(data_path, f"local_training_{self.client_id}.json")
        self.eval_data_path = os.path.join(data_path, f"local_eval_{self.client_id}.json")
        self.test_data_path = os.path.join(data_path, f"local_test_{self.client_id}.json")

        self.local_data = load_dataset("json", data_files=self.local_data_path, cache_dir=cache_dir)
        self.eval_data = load_dataset("json", data_files=self.eval_data_path, cache_dir=cache_dir)
        self.test_data = load_dataset("json", data_files=self.test_data_path, cache_dir=cache_dir)

        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", f"local_output_{self.client_id}")

        self.train_on_inputs = train_on_inputs
        self.cutoff_len = cutoff_len
        self.hetero_lora = hetero_lora
        self.optim = optim
        self.dataloader_num_workers = dataloader_num_workers
        self.pin_memory = torch.cuda.is_available()

        # eval knobs
        self.eval_protocol = (eval_protocol or "auto").lower()
        self.eval_answer_only_loss = bool(eval_answer_only_loss)
        self.eval_max_samples = int(eval_max_samples or 0)
        self.eval_gen_batch_size = int(eval_gen_batch_size or 1)
        self.eval_gen_max_new_tokens = int(eval_gen_max_new_tokens or 128)

    # ---- tokenization (with optional prompt masking) ----
    def generate_and_tokenize_prompt(self, data_point, mask_inputs: Optional[bool] = None):
        """
        mask_inputs:
          - None: follow training default (mask if not train_on_inputs)
          - True:  answer-only loss (mask prompt)
          - False: include prompt in loss
        """
        if mask_inputs is None:
            mask_inputs = (not self.train_on_inputs)

        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(self.tokenizer, full_prompt, cutoff_len=self.cutoff_len, add_eos_token=True)

        if mask_inputs:
            user_prompt = self.prompter.generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt = self.tokenizer(
                user_prompt,
                truncation=True,
                max_length=self.cutoff_len,
                padding=False,
                return_tensors=None,
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            )
        return tokenized_full_prompt

    def preprare_local_dataset(self, local_val_set_size=0):
        # Train: follow train_on_inputs
        train_map_fn = lambda x: self.generate_and_tokenize_prompt(x, mask_inputs=None)

        # Eval/Test: optionally force answer-only loss (community-friendly)
        eval_mask = True if self.eval_answer_only_loss else None
        eval_map_fn = lambda x: self.generate_and_tokenize_prompt(x, mask_inputs=eval_mask)

        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = local_train_val["train"].shuffle().map(train_map_fn)
            self.local_eval_dataset = local_train_val["test"].shuffle().map(eval_map_fn)
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(train_map_fn)
            self.local_eval_dataset = self.eval_data["train"].shuffle().map(eval_map_fn)
            self.local_test_dataset = self.test_data["train"].shuffle().map(eval_map_fn)

        self.local_val_set_size = len(self.local_eval_dataset)

    # ---- local trainer (training-time metrics can stay token-acc; we will use gen-metrics in test()) ----
    def build_local_trainer(
        self,
        tokenizer,
        local_micro_batch_size,
        gradient_accumulation_steps,
        local_num_epochs,
        local_learning_rate,
        group_by_length,
        warmup=0,
        lambd=None,
        reg=None,
    ):
        def compute_metrics(pred):
            # Keep a lightweight token-level metric during training (optional).
            logits = pred.predictions
            if isinstance(logits, tuple):
                logits = logits[0]
            labels_ids = pred.label_ids
            pred_ids = np.argmax(logits, axis=-1)
            shift_preds = pred_ids[:, :-1]
            shift_labels = labels_ids[:, 1:]
            mask = (shift_labels != -100)
            matches = (shift_preds == shift_labels) & mask
            correct = matches.sum()
            total_valid_tokens = mask.sum()
            accuracy = correct / max(1, total_valid_tokens)
            return {"accuracy": round(float(accuracy), 4)}

        use_cuda = torch.cuda.is_available()
        major, _ = torch.cuda.get_device_capability(0) if use_cuda else (0, 0)
        use_bf16 = use_cuda and major >= 8

        self.train_args = make_training_arguments(
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

        self.local_trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            optimizers=(optimizer, scheduler),
            compute_metrics=compute_metrics,
        )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if "lora" in name)
        )
        self.params_dict_new = OrderedDict(
            (name, param.detach()) for name, param in self.model.named_parameters() if "lora" in name
        )
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(instance, self.params_dict_new, "lora")
        ).__get__(self.model, type(self.model))

    def train(self):
        result = self.local_trainer.train()
        logging.info(self.local_trainer.state.log_history[-2])
        logging.info(self.local_trainer.state.log_history[-1])
        logging.info(result.metrics)
        return self.local_trainer.state.log_history[-2]

    # -----------------------------
    # Generation-based evaluation (community metrics)
    # -----------------------------
    def _infer_dataset_tag(self) -> str:
        # data_path/.../local_eval_{id}.json -> infer tag from parent dirs is hard here.
        # Instead rely on category field if exists; otherwise fallback to empty.
        try:
            ex0 = self.eval_data["train"][0]
            return str(ex0.get("category", "")).lower()
        except Exception:
            return ""

    def _get_task_type(self, dataset_hint: str) -> str:
        """
        Return one of: mc, math, nlg, unknown
        dataset_hint: either dataset_tag (from main) or category field
        """
        h = (dataset_hint or "").lower()
        if h in {"hellaswag", "piqa", "winogrande", "boolq"}:
            return "mc"
        if h in {"gsm8k", "metamathqa", "arithmetic", "svamp"}:
            return "math"
        if h in {"e2e_nlg", "e2e", "alpaca"}:
            return "nlg"
        return "unknown"

    def _build_prompts_and_refs(self, split: str = "eval") -> Tuple[List[str], List[Dict]]:
        raw = self.eval_data["train"] if split == "eval" else self.test_data["train"]
        items = []
        n = len(raw)
        if self.eval_max_samples and self.eval_max_samples > 0:
            n = min(n, self.eval_max_samples)
        for i in range(n):
            items.append(raw[i])
        prompts = [self.prompter.generate_prompt(x["instruction"], x["input"]) for x in items]
        return prompts, items

    def _generate_batch(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        self.model.eval()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        device = next(self.model.parameters()).device
        outs: List[str] = []

        bs = max(1, self.eval_gen_batch_size)
        for s in range(0, len(prompts), bs):
            chunk = prompts[s:s+bs]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cutoff_len,
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            lengths = attn.sum(dim=1).tolist()

            with torch.no_grad():
                gen_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            for i in range(gen_ids.size(0)):
                gen_cont = gen_ids[i, int(lengths[i]):]
                outs.append(self.tokenizer.decode(gen_cont, skip_special_tokens=True))

        return outs

    def evaluate_generation(self, split: str = "eval", dataset_tag: Optional[str] = None) -> Dict[str, float]:
        prompts, items = self._build_prompts_and_refs(split=split)
        hint = (dataset_tag or "").lower() or self._infer_dataset_tag()
        task_type = self._get_task_type(hint)

        # per-task decode length
        if task_type == "mc":
            max_new = 4
        elif task_type == "math":
            max_new = max(256, self.eval_gen_max_new_tokens)
        elif task_type == "nlg":
            max_new = self.eval_gen_max_new_tokens
        else:
            max_new = self.eval_gen_max_new_tokens

        preds = self._generate_batch(prompts, max_new_tokens=max_new)

        # ---- metrics ----
        if task_type == "mc":
            correct = 0
            total = max(1, len(items))
            for ptxt, ex in zip(preds, items):
                pred_lab = extract_mc_label(ptxt)
                gold = str(ex.get("output", "")).strip().upper()
                gold = "TRUE" if gold.lower() == "true" else "FALSE" if gold.lower() == "false" else gold
                if pred_lab == gold:
                    correct += 1
            return {"eval_accuracy": round(correct / total, 4)}

        if task_type == "math":
            correct = 0
            total = max(1, len(items))
            for ptxt, ex in zip(preds, items):
                cat = str(ex.get("category", "")).lower()
                gold_text = str(ex.get("output", ""))
                # Prefer a precomputed final_answer field if you add it later
                gold_final = str(ex.get("final_answer", "")).strip()
                if gold_final:
                    gold_final = _normalize_math_token(gold_final)
                else:
                    if cat == "gsm8k":
                        gold_final = extract_gsm8k_final(gold_text)
                    else:
                        gold_final = extract_metamath_final(gold_text)

                if cat == "gsm8k":
                    pred_final = extract_gsm8k_final(ptxt)
                else:
                    pred_final = extract_metamath_final(ptxt)

                if pred_final and gold_final and pred_final == gold_final:
                    correct += 1
            return {"eval_em": round(correct / total, 4)}

        if task_type == "nlg":
            pred_str = [_norm_space(x) for x in preds]
            ref_str = [_norm_space(str(ex.get("output", ""))) for ex in items]

            # Use local rouge metric path as in your repo
            rouge = evaluate.load('./evaluate/metrics/rouge/rouge.py')
            rouge_out = rouge.compute(predictions=pred_str, references=ref_str, use_aggregator=True)
            out = {
                "eval_rouge1": round(float(rouge_out.get("rouge1", 0.0)), 4),
                "eval_rouge2": round(float(rouge_out.get("rouge2", 0.0)), 4),
                "eval_rougeL": round(float(rouge_out.get("rougeL", 0.0)), 4),
                "eval_rougeLsum": round(float(rouge_out.get("rougeLsum", 0.0)), 4),
            }
            return out

        # fallback: return nothing
        return {}

    # -----------------------------
    # test(): eval_loss + generation-based metrics
    # -----------------------------
    def test(self, epoch, local_micro_batch_size, dataset_tag: Optional[str] = None):
        use_cuda = torch.cuda.is_available()
        major, _ = torch.cuda.get_device_capability(0) if use_cuda else (0, 0)
        use_bf16 = use_cuda and major >= 8

        test_args = make_training_arguments(
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


        tester = transformers.Trainer(
            model=self.model,
            args=test_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )

        eval_results = tester.evaluate(self.local_eval_dataset)
        # Ensure key exists
        if "eval_loss" in eval_results:
            eval_results["eval_loss"] = float(eval_results["eval_loss"])

        # Add generation-based community metrics
        proto = self.eval_protocol
        if proto == "auto" or proto == "gen":
            try:
                gen_metrics = self.evaluate_generation(split="eval", dataset_tag=dataset_tag)
                eval_results.update(gen_metrics)
            except Exception as e:
                logging.warning(f"[Client {self.client_id}] generation eval failed: {e}")

        # legacy_tf kept for debugging; not used for community standard
        logging.info(f'For client {self.client_id}, eval result: {eval_results}')
        return eval_results

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        lora_params = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name and param.requires_grad:
                lora_params[name] = param
        single_output_dir = os.path.join(self.output_dir, str(self.client_id), f"local_output_epoch_{epoch}")
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(lora_params, single_output_dir + "/pytorch_model.bin")

        _ = self.model.load_state_dict(self.params_dict_old, strict=False)
        previously_selected_clients_set = previously_selected_clients_set | {self.client_id}
        last_client_id = self.client_id
        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id
