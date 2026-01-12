# 🚀 LLM Fine-tuning on Google Colab

This guide shows you how to fine-tune a language model to intelligently blend financial forecasts and provide reasoning.

## 📋 Prerequisites

1. ✅ You've successfully run the main pipeline and have artifacts with `llm_train.jsonl` and `llm_val.jsonl`
2. ✅ Google Colab account with GPU access (free tier works, but Pro+ is recommended)
3. ✅ Basic familiarity with Google Colab notebooks

## 🎯 What This Does

The LLM stacking approach:

- Takes ARIMA naive, drift, and XGBoost predictions as input
- Learns to intelligently combine them based on historical patterns
- Provides human-readable reasoning for each forecast
- Creates audit-friendly memos that match FinDeep's transparency goals

## 📁 Step 1: Prepare Your Files

### Option A: Upload Manually

1. Zip your entire `Findeep` project folder
2. Upload the zip to Google Colab
3. Extract in Colab

### Option B: Use Google Drive

1. Upload your `Findeep` folder to Google Drive
2. Mount Drive in Colab
3. Navigate to your project folder

## 💻 Step 2: Colab Notebook Setup

Create a new Google Colab notebook and run these cells:

### Cell 1: Check GPU and Setup

```python
# Check if GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU detected. Go to Runtime > Change runtime type > GPU")
```

### Cell 2: Navigate to Project and Install Requirements

```python
# If uploaded as zip
%cd /content
!unzip -q findeep_project.zip  # adjust filename as needed

# If using Google Drive (uncomment and modify path)
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/path/to/Findeep

# Navigate to project directory
%cd /content/Findeep  # adjust path as needed

# Install requirements
!pip install -r colab_requirements.txt
```

### Cell 3: Verify Training Data

```python
# Check that training data exists
import os
import json

artifacts_dir = "./artifacts"
train_file = f"{artifacts_dir}/llm_train.jsonl"
val_file = f"{artifacts_dir}/llm_val.jsonl"

if os.path.exists(train_file):
    with open(train_file) as f:
        train_samples = sum(1 for _ in f)
    print(f"✅ Training data found: {train_samples} samples")

    # Show a sample
    with open(train_file) as f:
        sample = json.loads(f.readline())
    print("\\nSample training data:")
    print(json.dumps(sample, indent=2)[:500] + "...")
else:
    print(f"❌ No training data found at {train_file}")
    print("Make sure you've run the main pipeline first!")

if os.path.exists(val_file):
    with open(val_file) as f:
        val_samples = sum(1 for _ in f)
    print(f"✅ Validation data found: {val_samples} samples")
else:
    print("⚠️ No validation data found")
```

### Cell 4: Start Fine-tuning

```python
# Run the fine-tuning script
!python colab_llm_finetune.py
```

### Cell 5: Monitor Progress (Optional)

```python
# If you want to monitor training with tensorboard
%load_ext tensorboard
%tensorboard --logdir ./llm_ts_qlora/logs
```

## 📊 Step 3: Expected Training Results

### Training Duration:

- **Free Colab GPU (T4)**: ~30-60 minutes for 3 epochs
- **Colab Pro (V100/A100)**: ~15-30 minutes for 3 epochs

### Memory Usage:

- **Mistral-7B with QLoRA**: ~12-15GB VRAM
- **Alternative models**: DialoGPT-medium (~2-4GB), GPT-2 medium (~1-2GB)

### Training Output:

```
Training complete. LoRA adapters saved in ./llm_ts_qlora
```

## 🎯 Step 4: Test Your Fine-tuned Model

### Cell 6: Load and Test Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# Load base model and tokenizer
base_model = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load fine-tuned adapters
model = PeftModel.from_pretrained(model, "./llm_ts_qlora")

print("✅ Fine-tuned model loaded successfully!")

# Test with sample data
test_input = """<system>
You are a financial forecasting assistant. Combine baseline predictions and tiny history to forecast the next period and explain briefly.
</system>

<user>
{"company": "Tesla", "kpi": "Revenues", "history": [{"t": -2, "value": 25182000000.0}, {"t": -1, "value": 22496000000.0}], "baselines": {"arima_naive": 22496000000.0, "arima_drift": 20800000000.0, "xgb_point": 24500000000.0}, "exog": {"OperatingExpenses": 3200000000.0, "GrossProfit": 4200000000.0}}
</user>

<assistant>
"""

# Tokenize and generate
inputs = tokenizer(test_input, return_tensors="pt", truncation=True, max_length=1024)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\\n=== LLM Forecast ===")
print(response[len(test_input):])
```

## 📁 Step 5: Download Your Model

```python
# Zip the trained model for download
!zip -r llm_ts_qlora.zip ./llm_ts_qlora/

# Download via Colab
from google.colab import files
files.download('llm_ts_qlora.zip')
```

## 🔧 Troubleshooting

### Common Issues:

1. **Out of Memory Error**:

   ```python
   # Reduce batch size in colab_llm_finetune.py
   per_device_train_batch_size=1  # instead of 2
   gradient_accumulation_steps=16  # instead of 8
   ```

2. **No Training Data**:

   - Make sure you've run `python run_eval.py` first
   - Check that `artifacts/llm_train.jsonl` exists

3. **Slow Training**:

   - Verify GPU is enabled: Runtime > Change runtime type > Hardware accelerator > GPU
   - Use Colab Pro for faster GPUs

4. **Model Loading Errors**:
   - Some models require Hugging Face authentication
   - Alternative: Replace with `"microsoft/DialoGPT-medium"` for easier setup

## 🎯 Business Value

Once trained, your LLM will:

- **Intelligently blend** multiple forecasting models
- **Provide reasoning** for each prediction
- **Generate audit trails** for regulatory compliance
- **Adapt to company-specific patterns** through fine-tuning
- **Scale to new companies** without retraining baseline models

## 📈 Next Steps

1. **Experiment with different base models** (GPT-2, DialoGPT, Llama)
2. **Adjust LoRA parameters** for better performance
3. **Collect more training data** from additional companies
4. **Deploy the model** for production forecasting
5. **Create automated reporting** combining predictions with reasoning

Your fine-tuned LLM will now serve as an intelligent financial analyst that can blend quantitative models with qualitative insights! 🚀

## 🔧 Quick Start: 2023 Tesla-only Finetune (pre-2023 train, 2023 eval)

Use this if you uploaded the 2023-specific bundle `findeep_llm_2023_colab.zip`.

### Cell A: Unzip and install requirements

```python
# Adjust filename if needed
%cd /content
!unzip -q findeep_llm_2023_colab.zip

# Install requirements in Colab runtime
!pip install -r colab_requirements.txt
```

### Cell B: (Optional) Choose a different base model

```python
# Default is HuggingFaceTB/SmolLM2-1.7B-Instruct; you can switch here
import os
os.environ["BASE_MODEL"] = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
```

### Cell C: Run the 2023 workflow

```python
# This will:
# - create artifacts/llm_train_pre2023.jsonl and artifacts/llm_val_2023.jsonl
# - train QLoRA adapters into ./llm_ts_qlora
# - save 2023 Q1..Q4 JSON outputs under slide_assets/llm_outputs_2023/
!python colab_llm_finetune_tesla_2023.py
```

### Cell D (Optional): Inspect saved outputs

```python
import os, json, glob

print("Adapters folder:", os.path.exists("llm_ts_qlora"))
files = sorted(glob.glob("slide_assets/llm_outputs_2023/*.json"))
print("Saved 2023 outputs:", files)
if files:
    with open(files[0], "r", encoding="utf-8") as f:
        print("\nSample output:\n", f.read()[:600])
```

### Cell E (Optional): Download adapters and outputs

```python
# Zip the trained adapters and JSON outputs for download
!zip -rq llm_ts_qlora_final.zip llm_ts_qlora/
!zip -rq llm_outputs_2023.zip slide_assets/llm_outputs_2023/

from google.colab import files
files.download('llm_ts_qlora_final.zip')
files.download('llm_outputs_2023.zip')
```

Notes:

- The script prefers `artifacts/predictions.parquet` to build pre-2023 train and exact 2023Q1..Q4 validation examples. If missing, it falls back to `artifacts/llm_train.jsonl` (no year split).
- Adapters are saved under `llm_ts_qlora/` with a `final_adapter/` for easy loading later.
