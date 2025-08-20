import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


_MODEL = None
_TOKENIZER = None


MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def get_model_and_tokenizer():
global _MODEL, _TOKENIZER
if _MODEL is None or _TOKENIZER is None:
_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
_MODEL = AutoModelForCausalLM.from_pretrained(
MODEL_ID,
torch_dtype=DTYPE,
device_map="auto",
)
return _MODEL, _TOKENIZER
