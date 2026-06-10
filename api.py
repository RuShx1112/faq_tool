"""
Optional: REST API wrapper around FAQBot

Run:
    python api.py

Endpoints:
    POST /ask
        Body: {"question": "Is IVF painful?"}
        Returns: {"answer": "...", "sources": [...]}

    GET /health
        Returns: {"status": "ok"}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from flask import Flask, jsonify, request, render_template
from rag_engine import FAQBot

DATA_PATH = Path(__file__).parent / "data" / "faqs.json"

app = Flask(__name__)

print("Loading FAQ index…")
_bot = FAQBot(DATA_PATH)
print("Ready.")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/", methods=["GET", "POST"])
def home():

    answer = None

    if request.method == "POST":

        question = request.form.get("question", "")

        if question:
            result = _bot.ask(question)
            answer = result["answer"]

    return render_template(
        "index.html",
        answer=answer
    )

@app.post("/ask")
def ask():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    result = _bot.ask(question)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)


