import subprocess, sys

print("Evaluating...")
subprocess.check_call([sys.executable, "src/make_llm_dataset.py"])  # build LLM jsonl too
subprocess.check_call([sys.executable, "src/eval_compare.py"])
print("Check ./artifacts/metrics.csv and ./artifacts/ for outputs.")