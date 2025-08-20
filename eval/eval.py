from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_summarize_contract():
payload = {"text": "This is a long enough text for testing the summarizer. "*5, "style":"formal"}
r = client.post("/summarize", json=payload)
assert r.status_code == 200
js = r.json()
assert js["style"] == "formal"
assert isinstance(js["summary"], str) and len(js["summary"]) > 0
