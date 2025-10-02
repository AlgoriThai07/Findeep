import subprocess, sys

# 1) prepare data
print("[1/2] Preparing data...")
subprocess.check_call([sys.executable, "src/data_prep.py"])

# 2) train baselines + predict
print("[2/2] Training baselines and predicting...")
subprocess.check_call([sys.executable, "src/baselines.py"])

print("Done. Artifacts saved in ./artifacts/")