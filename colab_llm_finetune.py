# finetune.py
# QLoRA SFT for stacking on Colab GPU
# Steps in Colab:
#   pip install -r colab_requirements.txt
#   python finetune.py

import os
import json
import yaml
import torch
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Optional: silence wandb if you don't use it
os.environ.setdefault("WANDB_DISABLED", "true")

# ----------------------------
# Config & paths
# ----------------------------
CFG_PATH = "config.yaml"
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

ARTI_DIR = cfg.get("artifacts_dir", "./artifacts")
TRAIN_PATH = os.path.join(ARTI_DIR, "llm_train.jsonl")
VAL_PATH   = os.path.join(ARTI_DIR, "llm_val.jsonl")

if not os.path.exists(TRAIN_PATH) or os.path.getsize(TRAIN_PATH) == 0:
    raise FileNotFoundError(
        f"Training file missing or empty: {TRAIN_PATH}.\n"
        "Create it via `python run_eval.py` (or `python src/make_llm_dataset.py`)."
    )

# ----------------------------
# Data loader (JSONL -> 'text' column)
# ----------------------------
def load_jsonl_as_text_dataset(path: str):
    import datasets as ds
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj["messages"]
            chunks = []
            for m in msgs:
                role = m["role"]
                content = m["content"]
                if isinstance(content, dict):
                    content = json.dumps(content, ensure_ascii=False)
                chunks.append(f"<{role}>\n{content}\n</{role}>")
            rows.append({"text": "\n".join(chunks)})
    return datasets_from_list(rows=rows)

def datasets_from_list(rows):
    import datasets as ds
    return ds.Dataset.from_list(rows)

train_ds = load_jsonl_as_text_dataset(TRAIN_PATH)
val_ds = load_jsonl_as_text_dataset(VAL_PATH) if os.path.exists(VAL_PATH) and os.path.getsize(VAL_PATH) > 0 else None

print(f"Train examples: {len(train_ds)}")
print(f"Val examples:   {len(val_ds) if val_ds is not None else 0}")
if len(train_ds) == 0:
    raise ValueError("Your training dataset has 0 rows. Add more data or rebuild JSONL.")

# ----------------------------
# Model choice (open, no gate)
# ----------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")  # change if you want (e.g., Qwen/Qwen2.5-3B-Instruct)

# 4-bit QLoRA quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for 4-bit training (critical!)
model = prepare_model_for_kbit_training(model)

# ----------------------------
# Attach LoRA adapters (auto-detect targets)
# ----------------------------
primary_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
fallback_targets = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "Wqkv", "out_proj", "fc1", "fc2"]

def attach_lora(m, targets):
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=targets, bias="none", task_type="CAUSAL_LM"
    )
    return get_peft_model(m, peft_cfg)

def print_trainable(m):
    trainable, total = 0, 0
    for p in m.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = (100.0 * trainable / total) if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")

# Try primary targets
model = attach_lora(model, primary_targets)
print("After attaching primary targets:")
print_trainable(model)

if not any(p.requires_grad for p in model.parameters()):
    print("No trainable params with primary targets; retrying fallback target names…")
    # Reload fp4 base quickly, re-prepare, and re-attach with fallback targets
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = attach_lora(model, fallback_targets)
    print("After attaching fallback targets:")
    print_trainable(model)

# Good practice with checkpointing
try:
    model.config.use_cache = False
except Exception:
    pass
model.train()

# ----------------------------
# GPU-aware defaults
# ----------------------------
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print("GPU:", gpu_name)

# Defaults safe for T4/V100; scale up on A100/L4
per_device_bs, grad_accum, max_seq_len, use_bf16 = 1, 16, 2048, True
if "A100" in gpu_name:
    per_device_bs, grad_accum = 4, 4
elif "L4" in gpu_name:
    per_device_bs, grad_accum = 2, 8
elif "V100" in gpu_name:
    per_device_bs, grad_accum = 1, 16
elif "T4" in gpu_name:
    per_device_bs, grad_accum = 1, 16

print(f"Using batch_size={per_device_bs}, grad_accum={grad_accum}, max_seq_length={max_seq_len}, bf16={use_bf16}")

# ----------------------------
# Trainer (packing OFF for tiny datasets)
# ----------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tok,  # TRL 0.9.6 accepts tokenizer=
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=SFTConfig(
        output_dir="./llm_ts_qlora",
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=50,
        save_steps=200,
        bf16=use_bf16,
        gradient_checkpointing=True,
        packing=False,              # keep OFF for small datasets
        dataset_text_field="text",  # dataset provides a 'text' column
        max_seq_length=max_seq_len,
    ),
)

print("Model in train mode:", model.training)
print("Any param requires_grad?:", any(p.requires_grad for p in model.parameters()))

trainer.train()
print("Training complete. LoRA adapters saved in ./llm_ts_qlora")

# Optional: evaluate if val set exists
if val_ds is not None and len(val_ds) > 0:
    print("Evaluating on validation set...")
    metrics = trainer.evaluate()
    print(metrics)
