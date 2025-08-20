import argparse, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from training.data_utils import load_jsonl, format_examples
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="training/config.yaml")
parser.add_argument("--data", default="data/samples.jsonl")
parser.add_argument("--style", default="formal", choices=["formal","casual","bullet"])
args = parser.parse_args()


cfg = yaml.safe_load(open(args.config))
model_id = cfg["model_id"]
qlora = cfg.get("qlora", {}).get("use_qlora", False)


print(f"Loading {model_id} (QLoRA={qlora})")


# tokenizer
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)


# model
if qlora:
from transformers import BitsAndBytesConfig
bnb_cfg = BitsAndBytesConfig(
load_in_4bit=cfg["qlora"]["load_in_4bit"],
bnb_4bit_use_double_quant=cfg["qlora"]["bnb_4bit_use_double_quant"],
bnb_4bit_quant_type=cfg["qlora"]["bnb_4bit_quant_type"],
bnb_4bit_compute_dtype=torch.float16,
)
base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_cfg, device_map="auto")
base = prepare_model_for_kbit_training(base)
else:
base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto")


lora_cfg = LoraConfig(
r=cfg["lora"]["r"],
lora_alpha=cfg["lora"]["alpha"],
lora_dropout=cfg["lora"]["dropout"],
target_modules=cfg["lora"]["target_modules"],
bias="none",
task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora_cfg)


# data
raw = load_jsonl(args.data)
proc = format_examples(raw, args.style)


def tokenize(batch):
out = tok(batch["input_text"], truncation=True, max_length=cfg["train"]["max_seq_length"])
with tok.as_target_tokenizer():
labels = tok(batch["labels"], truncation=True, max_length=cfg["train"]["max_seq_length"])['input_ids']
out["labels"] = labels
return out


proc = proc.map(tokenize, batched=True, remove_columns=proc.column_names)


args_tr = TrainingArguments(
output_dir=cfg["output_dir"],
per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
learning_rate=cfg["train"]["learning_rate"],
num_train_epochs=cfg["train"]["num_train_epochs"],
logging_steps=cfg["train"]["logging_steps"],
save_steps=cfg["train"]["save_steps"],
bf16=cfg["train"].get("bf16", False),
warmup_ratio=cfg["train"].get("warmup_ratio", 0.05),
lr_scheduler_type=cfg["train"].get("lr_scheduler_type", "cosine"),
)


collator = DataCollatorForLanguageModeling(tok, mlm=False)


trainer = Trainer(
model=model,
args=args_tr,
train_dataset=proc,
data_collator=collator,
)


print("Saved LoRA adapter.")
