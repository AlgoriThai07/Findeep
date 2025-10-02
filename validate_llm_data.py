#!/usr/bin/env python3
"""
Local test script for LLM training data format
Run this before uploading to Colab to verify your data is properly formatted
"""

import json
import os
from pathlib import Path

def validate_training_data():
    """Validate that training data is properly formatted"""
    
    artifacts_dir = "./artifacts"
    train_file = f"{artifacts_dir}/llm_train.jsonl"
    val_file = f"{artifacts_dir}/llm_val.jsonl"
    
    print("=== LLM Training Data Validation ===")
    
    # Check if files exist
    if not os.path.exists(train_file):
        print(f"❌ Training file not found: {train_file}")
        print("Run the main pipeline first: python run_eval.py")
        return False
    
    # Load and validate training data
    train_samples = []
    try:
        with open(train_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    train_samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON on line {line_num}: {e}")
                    return False
    except Exception as e:
        print(f"❌ Error reading training file: {e}")
        return False
    
    print(f"✅ Training data: {len(train_samples)} samples loaded")
    
    # Validate sample structure
    if train_samples:
        sample = train_samples[0]
        required_keys = ['messages']
        
        for key in required_keys:
            if key not in sample:
                print(f"❌ Missing key '{key}' in training sample")
                return False
        
        # Check messages structure
        messages = sample['messages']
        if not isinstance(messages, list):
            print("❌ 'messages' should be a list")
            return False
        
        role_count = {'system': 0, 'user': 0, 'assistant': 0}
        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                print("❌ Each message needs 'role' and 'content'")
                return False
            role_count[msg['role']] = role_count.get(msg['role'], 0) + 1
        
        print(f"✅ Message structure valid: {role_count}")
        
        # Show sample
        print("\n=== Sample Training Data ===")
        print(json.dumps(sample, indent=2)[:800] + "...")
    
    # Check validation data if exists
    val_samples = []
    if os.path.exists(val_file):
        try:
            with open(val_file, 'r') as f:
                for line in f:
                    val_samples.append(json.loads(line.strip()))
            print(f"✅ Validation data: {len(val_samples)} samples loaded")
        except Exception as e:
            print(f"⚠️  Warning: Error reading validation file: {e}")
    else:
        print("⚠️  No validation data found (optional)")
    
    # Summary
    total_samples = len(train_samples) + len(val_samples)
    print(f"\n=== Summary ===")
    print(f"Total samples for LLM training: {total_samples}")
    print(f"Training/Validation split: {len(train_samples)}/{len(val_samples)}")
    
    if total_samples < 5:
        print("⚠️  Warning: Very few samples. Consider adding more companies or time periods.")
    elif total_samples < 20:
        print("⚠️  Limited samples. Training may overfit. Consider data augmentation.")
    else:
        print("✅ Good amount of training data!")
    
    return True

def estimate_training_time():
    """Estimate training time and requirements"""
    
    print("\n=== Training Estimates ===")
    print("Model: Mistral-7B with QLoRA")
    print("Hardware requirements:")
    print("  - Minimum: 12GB VRAM (Colab T4)")
    print("  - Recommended: 16GB+ VRAM (Colab Pro V100/A100)")
    print()
    print("Estimated training time (3 epochs):")
    print("  - Colab Free (T4): 45-90 minutes")
    print("  - Colab Pro (V100): 20-40 minutes") 
    print("  - Colab Pro+ (A100): 10-20 minutes")
    print()
    print("Alternative lighter models:")
    print("  - microsoft/DialoGPT-medium: ~5-15 minutes")
    print("  - gpt2-medium: ~3-10 minutes")

def create_colab_zip():
    """Create a zip file ready for Colab upload"""
    
    import shutil
    import zipfile
    
    print("\n=== Creating Colab Upload Package ===")
    
    # Files to include
    essential_files = [
        "colab_llm_finetune.py",
        "colab_requirements.txt", 
        "config.yaml",
        "artifacts/llm_train.jsonl",
        "artifacts/llm_val.jsonl",
        "COLAB_LLM_SETUP.md"
    ]
    
    optional_files = [
        "artifacts/predictions.parquet",
        "artifacts/metrics.csv",
        "filtered_combined.json"
    ]
    
    zip_name = "findeep_llm_colab.zip"
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add essential files
        for file_path in essential_files:
            if os.path.exists(file_path):
                zipf.write(file_path)
                print(f"✅ Added: {file_path}")
            else:
                print(f"⚠️  Missing: {file_path}")
        
        # Add optional files
        for file_path in optional_files:
            if os.path.exists(file_path):
                zipf.write(file_path)
                print(f"✅ Added (optional): {file_path}")
    
    zip_size = os.path.getsize(zip_name) / (1024*1024)  # MB
    print(f"\n✅ Created {zip_name} ({zip_size:.1f} MB)")
    print(f"Ready to upload to Google Colab!")
    
    return zip_name

if __name__ == "__main__":
    print("🤖 FinDeep LLM Training Validator\n")
    
    # Validate training data
    if validate_training_data():
        print("\n✅ Training data validation passed!")
        
        # Show estimates
        estimate_training_time()
        
        # Ask if user wants to create zip
        try:
            create_zip = input("\nCreate Colab upload zip? (y/n): ").lower().strip()
            if create_zip in ['y', 'yes']:
                zip_file = create_colab_zip()
                print(f"\nNext steps:")
                print(f"1. Upload {zip_file} to Google Colab")
                print(f"2. Follow instructions in COLAB_LLM_SETUP.md")
                print(f"3. Run the fine-tuning!")
        except KeyboardInterrupt:
            print("\nValidation complete!")
    
    else:
        print("\n❌ Training data validation failed!")
        print("Fix the issues above before proceeding to Colab.")