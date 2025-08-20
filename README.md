# 📝 LLM Text Summarizer (Multi‑Style Output)

This project is a **production‑like, portfolio‑ready LLM app** that generates summaries in different styles — **formal**, **casual**, and **bullet‑point** — via a REST API. It includes LoRA/QLoRA fine‑tuning, evaluation scripts, tests, Docker support, and CI hooks.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/Web-FastAPI-green)
![Transformers](https://img.shields.io/badge/HF-Transformers-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

* Multi‑style summaries (formal, casual, bullet‑point)
* Controllable prompts for different styles
* LoRA/QLoRA fine‑tuning on a custom dataset
* FastAPI inference server with REST endpoint
* Offline evaluation (ROUGE, BERTScore)
* Dockerized + GitHub Actions template

---

## 📂 Repository Structure

```
llm-summarizer/
├─ app/              # FastAPI app, prompt templates, schemas
├─ training/         # LoRA/QLoRA fine‑tuning scripts
├─ eval/             # Evaluation scripts
├─ tests/            # Unit & API tests
├─ docker/           # Dockerfile
├─ scripts/          # Utility scripts (adapter export)
├─ data/             # Dataset samples & format docs
├─ requirements.txt
└─ README.md
```

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the API

```bash
export MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test the Endpoint

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text":"Your long article text...","style":"formal","max_tokens":200}'
```

Example response:

```json
{
  "style": "formal",
  "summary": "This article explains..."
}
```

---

## 📊 Data Format

Dataset format (JSONL):

```jsonl
{"text": "Long text...", "summary_formal": "...", "summary_casual": "...", "summary_bullet": ["• ...", "• ..."]}
```

See `data/README.md` for details.

---

## 🧩 Fine‑Tuning

* **LoRA** and **QLoRA** supported via HuggingFace PEFT.
* Configurable with `training/config.yaml`.
* Example:

```bash
python training/train_lora.py --data data/samples.jsonl --style formal
```

---

## 📏 Evaluation

Evaluate summaries with ROUGE and BERTScore:

```bash
python eval/eval.py --pred eval/sample_eval.jsonl
```

---

## 🧪 Tests

Run tests with pytest:

```bash
pytest tests/
```

---

## 🐳 Docker

Build and run container:

```bash
docker build -t llm-summarizer -f docker/Dockerfile .
docker run -p 8000:8000 llm-summarizer
```

---

## 📈 Roadmap / Stretch Goals

* [ ] Add Streamlit demo UI
* [ ] Push fine‑tuned adapters to HuggingFace Hub
* [ ] Add CI/CD via GitHub Actions
* [ ] Multilingual summarization

---

## 📜 License

MIT License — free to use and modify.

---

## 📣 Credits

Built with ❤️ using:

* [HuggingFace Transformers](https://huggingface.co/transformers/)
* [PEFT / LoRA](https://huggingface.co/docs/peft/)
* [FastAPI](https://fastapi.tiangolo.com/)
