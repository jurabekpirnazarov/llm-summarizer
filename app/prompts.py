SYSTEM_PROMPT = """
You are a precise summarization assistant. Produce concise, faithful summaries that preserve key facts.
""".strip()


STYLE_TEMPLATES = {
"formal": "Summarize the text in a professional, concise tone. Limit to {max_sents} sentences.",
"casual": "Summarize the text in a friendly, approachable tone. Keep it simple and clear.",
"bullet": "Summarize the text as bullet points (3-6 items), each a short phrase.",
}


def build_prompt(style: str, text: str, max_sents: int = 3):
style_inst = STYLE_TEMPLATES.get(style, STYLE_TEMPLATES["formal"]).format(max_sents=max_sents)
return f"<s>[INST]\n{style_inst}\n\nText:\n{text}\n[/INST]</s>"
