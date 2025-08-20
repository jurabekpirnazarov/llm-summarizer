from app.prompts import build_prompt


def test_prompt_contains_text():
p = build_prompt("formal", "hello world")
assert "hello world" in p
