from fastapi import FastAPI, HTTPException
from app.schemas import SummarizeRequest, SummarizeResponse
from app.prompts import SYSTEM_PROMPT, build_prompt
from app.utils import get_model_and_tokenizer
import torch


app = FastAPI(title="LLM Summarizer (Multi-Style)")


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
if len(req.text.strip()) < 20:
raise HTTPException(status_code=422, detail="Text too short to summarize")


model, tok = get_model_and_tokenizer()
prompt = build_prompt(req.style, req.text)


inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
output = model.generate(
**inputs,
max_new_tokens=req.max_tokens,
temperature=req.temperature,
do_sample=True,
top_p=0.9,
eos_token_id=tok.eos_token_id,
)
full = tok.decode(output[0], skip_special_tokens=True)


# naive post-process: take last assistant segment
summary = full.split("[/INST]")[-1].strip()
return SummarizeResponse(style=req.style, summary=summary)
