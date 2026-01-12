#!/usr/bin/env python3
"""
Create a Colab-ready zip for the 2023 Tesla finetune script.

Includes:
 - colab_llm_finetune_tesla_2023.py
 - colab_requirements.txt
 - config.yaml
 - artifacts/predictions.parquet (primary source for splitting train/val)
 - COLAB_LLM_SETUP.md
Optional:
 - artifacts/llm_train.jsonl, artifacts/llm_val.jsonl (fallback / reference)

Output: findeep_llm_2023_colab.zip
"""

import os
import zipfile


def create_zip():
    # Prefer the 2023-specific parquet if available
    pred_2023 = os.path.join("artifacts", "predictions_2023.parquet")
    pred_legacy = os.path.join("artifacts", "predictions.parquet")
    parquet_to_use = pred_2023 if os.path.exists(pred_2023) else pred_legacy

    essentials = [
        "colab_llm_finetune_tesla_2023.py",
        "colab_requirements.txt",
        "config.yaml",
        "COLAB_LLM_SETUP.md",
        parquet_to_use,
    ]

    optional = [
        os.path.join("artifacts", "llm_train.jsonl"),
        os.path.join("artifacts", "llm_val.jsonl"),
    ]

    zip_name = "findeep_llm_2023_colab.zip"

    print("\n=== Creating 2023 Colab Upload Package ===")
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in essentials:
            if os.path.exists(p):
                zf.write(p)
                print(f"✅ Added: {p}")
            else:
                print(f"❌ Missing required file: {p}")
        for p in optional:
            if os.path.exists(p):
                zf.write(p)
                print(f"✅ Added (optional): {p}")

    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"\n✅ Created {zip_name} ({size_mb:.2f} MB)")
    print("Upload to Colab, unzip, pip install -r colab_requirements.txt, then run:")
    print("  python colab_llm_finetune_tesla_2023.py")
    return zip_name


if __name__ == "__main__":
    create_zip()
