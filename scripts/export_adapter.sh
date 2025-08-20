#!/usr/bin/env bash
BASE_MODEL=${1:-mistralai/Mistral-7B-Instruct-v0.2}
ADAPTER_DIR=${2:-outputs/lora/adapter-formal}
OUT_DIR=${3:-outputs/merged-formal}


python - <<'PY'
import sys
from transformers import AutoModelForCausalLM
from peft import PeftModel
base_id = sys.argv[1]
adapter_dir = sys.argv[2]
out_dir = sys.argv[3]
base = AutoModelForCausalLM.from_pretrained(base_id)
peft = PeftModel.from_pretrained(base, adapter_dir)
merged = peft.merge_and_unload()
merged.save_pretrained(out_dir)
print("Merged to", out_dir)
PY
"$BASE_MODEL" "$ADAPTER_DIR" "$OUT_DIR"
