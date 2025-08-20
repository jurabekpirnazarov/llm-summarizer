from datasets import load_dataset


def load_jsonl(path: str):
# expects a JSONL with keys: text, summary_formal/summary_casual/summary_bullet
return load_dataset("json", data_files=path, split="train")


def format_examples(ds, style: str):
key = {
"formal": "summary_formal",
"casual": "summary_casual",
"bullet": "summary_bullet",
}[style]


def _map(ex):
gold = ex.get(key)
if isinstance(gold, list):
gold = "\n".join(gold)
prompt = f"Summarize the text in {style} style.\n\nText:\n{ex['text']}\n"
return {"input_text": prompt, "labels": gold}


return ds.map(_map, remove_columns=[c for c in ds.column_names if c not in ["input_text","labels"]])
