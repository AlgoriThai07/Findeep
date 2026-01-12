# finetune_2023.py
# Train on pre-2023 data; predict & save JSON outputs for all 2023 quarters (Q1..Q4).
# Works across ALL companies/KPIs found in artifacts/predictions.parquet.

import os, re, glob, json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import yaml
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

# -----------------------------
# Basic setup
# -----------------------------
os.environ.setdefault("WANDB_DISABLED", "true")
torch.manual_seed(42)

CFG_PATH = "config.yaml"
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

ARTI_DIR = cfg.get("artifacts_dir", "./artifacts")
os.makedirs(ARTI_DIR, exist_ok=True)

PRED_2023 = os.path.join(ARTI_DIR, "predictions_2023.parquet")
PRED_LEGACY = os.path.join(ARTI_DIR, "predictions.parquet")
PRED_PATH = PRED_2023 if os.path.exists(PRED_2023) else PRED_LEGACY
TRAIN_JSONL = os.path.join(ARTI_DIR, "llm_train_pre2023.jsonl")
VAL_JSONL   = os.path.join(ARTI_DIR, "llm_val_2023.jsonl")

ADAPTER_DIR = "llm_ts_qlora"
FINAL_ADAPTER_DIR = os.path.join(ADAPTER_DIR, "final_adapter")
SLIDE_DIR = "slide_assets/llm_outputs_2023"
os.makedirs("slide_assets", exist_ok=True)
os.makedirs(SLIDE_DIR, exist_ok=True)

BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")  # small, open, ungated

# -----------------------------
# Helpers: year/quarter detection & column normalization
# -----------------------------
def add_year_quarter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Prefer explicit columns if present
    if {"year","quarter"}.issubset(df.columns):
        return df
    if {"target_year","target_quarter"}.issubset(df.columns):
        df["year"] = df["target_year"].astype(int)
        df["quarter"] = df["target_quarter"].astype(int)
        return df
    # Period like "2023Q1" or "2023-Q1"
    if "Period" in df.columns:
        def parse_period(p):
            m = re.search(r"(20\d{2})\s*[- ]?\s*Q([1-4])", str(p), re.IGNORECASE)
            return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)
        yq = df["Period"].apply(parse_period)
        df["year"] = yq.map(lambda t: t[0])
        df["quarter"] = yq.map(lambda t: t[1])
        if df["year"].notna().any():
            return df
    # Datetime fallback
    for c in ["target_date","report_date","date","asof_date"]:
        if c in df.columns:
            d = pd.to_datetime(df[c], errors="coerce")
            if d.notna().any():
                df["year"] = d.dt.year
                df["quarter"] = ((d.dt.month-1)//3 + 1).astype(int)
                return df
    raise ValueError("Could not infer year/quarter. Ensure predictions.parquet has either "
                     "'year'+'quarter', 'target_year'+'target_quarter', 'Period' like 2023Q1, or a target_date column.")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Company
    if "Company" not in df.columns:
        for c in ["company","ticker","firm"]:
            if c in df.columns:
                df = df.rename(columns={c: "Company"})
                break
    # KPI/metric
    if "KPI" not in df.columns:
        for c in ["kpi","metric"]:
            if c in df.columns:
                df = df.rename(columns={c: "KPI"})
                break
    # Actual
    if "actual" not in df.columns:
        for c in ["y_true","target","y"]:
            if c in df.columns:
                df = df.rename(columns={c: "actual"})
                break
    # Baselines
    if "arima_naive" not in df.columns:
        for c in ["naive","arima_last","last_value"]:
            if c in df.columns:
                df = df.rename(columns={c: "arima_naive"})
                break
    if "arima_drift" not in df.columns:
        for c in ["drift","arima_drift_forecast"]:
            if c in df.columns:
                df = df.rename(columns={c: "arima_drift"})
                break
    if "xgb_point" not in df.columns:
        for c in ["xgb","xgb_pred","xgboost","xgb_point_forecast"]:
            if c in df.columns:
                df = df.rename(columns={c: "xgb_point"})
                break

    # sanity
    for need in ["Company","KPI","year","quarter"]:
        if need not in df.columns:
            raise ValueError(f"Missing required column: {need}")
    for num in ["actual","arima_naive","arima_drift","xgb_point"]:
        if num not in df.columns:
            df[num] = np.nan
    return df

# -----------------------------
# Build JSONL (chat format)
# -----------------------------
def build_examples_for_series(sdf: pd.DataFrame, history_len: int = 2) -> List[Dict[str,Any]]:
    sdf = sdf.sort_values(["year","quarter"])
    rows = sdf.to_dict("records")
    out = []
    for i, cur in enumerate(rows):
        prior = rows[max(0, i-history_len):i]
        if not prior:
            continue
        history = [{"t": j-len(prior), "value": float(p.get("actual", np.nan))}
                   for j, p in enumerate(prior)]
        baselines = {}
        for k in ["arima_naive","arima_drift","xgb_point"]:
            v = cur.get(k, np.nan)
            if not pd.isna(v):
                baselines[k] = float(v)
        exog = {}
        for k in ["OperatingExpenses","GrossProfit","Assets","StockholdersEquity"]:
            if k in cur and not pd.isna(cur[k]):
                exog[k] = float(cur[k])

        system_msg = {
            "role": "system",
            "content": ("You are a financial forecasting assistant. Combine baseline predictions and tiny history "
                        "to forecast the next period and explain briefly. Always return a JSON object with keys "
                        "'final_forecast' (number) and 'explanation' (string).")
        }
        user_payload = {
            "company": cur["Company"], "kpi": cur["KPI"],
            "target": {"year": int(cur["year"]), "quarter": int(cur["quarter"])},
            "history": history, "baselines": baselines, "exog": exog
        }
        user_msg = {"role": "user", "content": user_payload}

        asst_msg = None
        if not pd.isna(cur.get("actual", np.nan)):
            asst_msg = {"role": "assistant",
                        "content": {"final_forecast": float(cur["actual"]),
                                    "explanation": "Ground truth for supervised SFT."}}
        out.append({"messages": [system_msg, user_msg] + ([asst_msg] if asst_msg else [])})
    return out

def split_train_val(df: pd.DataFrame, year_val: int = 2023) -> Tuple[List[Dict], List[Dict]]:
    df = add_year_quarter(df)
    df = normalize_columns(df)

    train, val = [], []
    for (co, kpi), sdf in df.groupby(["Company","KPI"]):
        exs = build_examples_for_series(sdf, history_len=2)
        for e in exs:
            tgt = e["messages"][1]["content"]["target"]
            y, q = int(tgt["year"]), int(tgt["quarter"])
            if y < year_val:
                train.append(e)
            elif y == year_val and q in {1,2,3,4}:
                # Validation examples for 2023; we keep assistant (actual) in the jsonl,
                # but at inference we’ll reassemble the user payload and ask the model to predict.
                val.append(e)
            # ignore y > 2023
    return train, val

def write_jsonl(path: str, rows: List[Dict[str,Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -----------------------------
# 1) Build train/val jsonl
# -----------------------------
if not os.path.exists(PRED_PATH):
    raise FileNotFoundError(
        "Missing artifacts/predictions_2023.parquet and artifacts/predictions.parquet. "
        "Run your baseline pipeline first (src/baselines_2023.py)."
    )

df = pd.read_parquet(PRED_PATH)
train_rows, val_rows = split_train_val(df, year_val=2023)

if not train_rows:
    raise ValueError("Training set is empty (no pre-2023 rows). Fill predictions.parquet with earlier periods.")
if not val_rows:
    print("WARNING: No 2023Q1..Q4 rows found for validation. The script will still train, but will skip 2023 inference.")

write_jsonl(TRAIN_JSONL, train_rows)
write_jsonl(VAL_JSONL, val_rows)
print(f"Wrote train: {TRAIN_JSONL}  ({len(train_rows)} rows)")
print(f"Wrote val:   {VAL_JSONL}    ({len(val_rows)} rows)")

# Convert JSONL -> HF Datasets with a single 'text' field (chat packed as plain text)
def jsonl_to_text_ds(path: str):
    import datasets as ds
    items = []
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks = []
            for m in obj["messages"]:
                if m is None: 
                    continue
                role, content = m["role"], m["content"]
                if isinstance(content, dict):
                    content = json.dumps(content, ensure_ascii=False)
                chunks.append(f"<{role}>\n{content}\n</{role}>")
            items.append({"text": "\n".join(chunks)})
    import datasets as ds
    return ds.Dataset.from_list(items) if items else None

train_ds = jsonl_to_text_ds(TRAIN_JSONL)
val_ds   = jsonl_to_text_ds(VAL_JSONL)

print("Train examples:", 0 if train_ds is None else len(train_ds))
print("Val examples:  ", 0 if val_ds is None else len(val_ds))

# -----------------------------
# 2) QLoRA fine-tuning
# -----------------------------
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

try:
    model.config.use_cache = False
except Exception:
    pass
model.train()

# Simple GPU-aware defaults
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
bsz, gas, max_len, bf16 = 1, 16, 2048, True
if "A100" in gpu: bsz, gas = 4, 4
elif "L4" in gpu: bsz, gas = 2, 8
print(f"GPU: {gpu} | batch={bsz} grad_accum={gas} max_len={max_len} bf16={bf16}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,   # TRL >= 0.9.6 accepts tokenizer=
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=SFTConfig(
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=gas,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=50,
        save_steps=200,
        bf16=bf16,
        gradient_checkpointing=True,
        packing=False,              # important for small datasets
        dataset_text_field="text",
        max_seq_length=max_len,
    ),
)

print("Start training…")
trainer.train()
print(f"Training complete. Adapters in: {ADAPTER_DIR}")

# Save a deterministic final adapter snapshot
os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)
trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
tok.save_pretrained(FINAL_ADAPTER_DIR)
print("Saved final adapters to:", FINAL_ADAPTER_DIR)

# -----------------------------
# 3) Auto-infer for all 2023 targets & save JSONs
# -----------------------------
def only_json_prompt(user_payload: Dict[str,Any]) -> str:
    return (
        "<system>\n"
        "You are a financial forecasting assistant. "
        "Output ONLY a JSON object with keys 'final_forecast' (number) and 'explanation' (short string). "
        "No extra text.\n"
        "</system>\n"
        f"<user>\n{json.dumps(user_payload, ensure_ascii=False)}\n</user>\n"
        "<assistant>\n"
    )

def run_infer_for_payloads(payloads: List[Dict[str,Any]], base_model: str, adapter_dir: str):
    # Load base + adapters for inference
    t = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if t.pad_token is None: t.pad_token = t.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto")
    m = PeftModel.from_pretrained(base, adapter_dir)
    m.eval()

    saved = []
    for p in payloads:
        co = p.get("company","Company"); kpi = p.get("kpi","KPI")
        tgt = p.get("target", {})
        y, q = tgt.get("year"), tgt.get("quarter")
        fname = f"{co}_{kpi}_{y}Q{q}.json".replace(" ","_")
        prompt = only_json_prompt(p)

        inputs = t(prompt, return_tensors="pt").to(m.device)
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=160, temperature=0.2, do_sample=False,
                             pad_token_id=t.eos_token_id, eos_token_id=t.eos_token_id)
        decoded = t.decode(out[0], skip_special_tokens=True)
        gen = decoded[len(prompt):].strip()

        # best-effort JSON parse
        try:
            ans = json.loads(gen)
        except json.JSONDecodeError:
            import re
            mobj = re.search(r"\{.*\}", gen, flags=re.DOTALL)
            if not mobj:
                ans = {"final_forecast": None, "explanation": gen[:200]}
            else:
                cand = re.sub(r",\s*([}\]])", r"\1", mobj.group(0))
                ans = json.loads(cand)

        with open(os.path.join(SLIDE_DIR, fname), "w", encoding="utf-8") as f:
            json.dump(ans, f, indent=2, ensure_ascii=False)
        saved.append(fname)
    return saved

# Collect user payloads for 2023 from our val jsonl (which may contain assistant ground truth too)
val_payloads = []
if os.path.exists(VAL_JSONL) and os.path.getsize(VAL_JSONL) > 0:
    with open(VAL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for m in obj["messages"]:
                if m and m.get("role") == "user":
                    val_payloads.append(m["content"])
                    break

if val_payloads:
    saved_files = run_infer_for_payloads(val_payloads, BASE_MODEL, FINAL_ADAPTER_DIR)
    print("Saved 2023 forecasts to:", SLIDE_DIR)
    for s in saved_files[:10]:
        print(" -", os.path.join(SLIDE_DIR, s))
    if len(saved_files) > 10:
        print(f" ... and {len(saved_files)-10} more")
else:
    print("No 2023 payloads to infer — skipping inference.")
