import json
import os

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

MODEL = "claude-haiku-4-5"
LANGUAGES = {"en-IN": "English", "te-IN": "Telugu", "hi-IN": "Hindi"}

app = Flask(__name__)
claude = (
    anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
        # Only needed when the key is not scoped to a workspace.
        default_headers=(
            {"anthropic-workspace-id": ANTHROPIC_WORKSPACE_ID}
            if ANTHROPIC_WORKSPACE_ID
            else None
        ),
    )
    if ANTHROPIC_API_KEY
    else None
)


def error(message, status):
    return jsonify(error=message), status


def call_claude(**kwargs):
    """Call Haiku; returns (response, None) or (None, JSON error response)."""
    try:
        return claude.messages.create(model=MODEL, **kwargs), None
    except anthropic.AuthenticationError:
        return None, error("Invalid Anthropic API key", 502)
    except anthropic.RateLimitError:
        return None, error("Claude API rate limit reached, try again shortly", 429)
    except anthropic.APITimeoutError:
        return None, error("Claude API timed out", 504)
    except anthropic.APIConnectionError:
        return None, error("Could not reach the Claude API", 502)
    except anthropic.APIStatusError as e:
        return None, error(f"Claude API error: {e.message}", 502)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/question", methods=["POST"])
def question():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error("Request body must be a JSON object", 400)

    topic = data.get("topic")
    language = data.get("language")
    previous = data.get("previous_questions", [])

    if not isinstance(topic, str) or not topic.strip():
        return error("'topic' must be a non-empty string", 400)
    if language not in LANGUAGES:
        return error("'language' must be one of: " + ", ".join(LANGUAGES), 400)
    if not isinstance(previous, list) or not all(isinstance(q, str) for q in previous):
        return error("'previous_questions' must be a list of strings", 400)
    if claude is None:
        return error("ANTHROPIC_API_KEY is not configured on the server", 500)

    asked = "\n".join(f"- {q}" for q in previous) or "(none)"
    prompt = (
        f"Write one new short factual quiz question about: {topic.strip()}\n"
        f"Language: {LANGUAGES[language]} ({language}).\n"
        "The question must be answerable in one spoken sentence, and must not "
        "repeat or closely rephrase any of these previously asked questions:\n"
        f"{asked}\n\n"
        "Reply with only the question text, nothing else."
    )

    response, err = call_claude(
        max_tokens=300, messages=[{"role": "user", "content": prompt}]
    )
    if err:
        return err

    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    if not text:
        return error("Claude returned no question", 502)
    return jsonify(question=text)


CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "feedback": {"type": "string"},
    },
    "required": ["correct", "feedback"],
    "additionalProperties": False,
}


@app.route("/check", methods=["POST"])
def check():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error("Request body must be a JSON object", 400)

    q = data.get("question")
    answer = data.get("answer")
    language = data.get("language")

    if not isinstance(q, str) or not q.strip():
        return error("'question' must be a non-empty string", 400)
    if not isinstance(answer, str) or not answer.strip():
        return error("'answer' must be a non-empty string", 400)
    if language not in LANGUAGES:
        return error("'language' must be one of: " + ", ".join(LANGUAGES), 400)
    if claude is None:
        return error("ANTHROPIC_API_KEY is not configured on the server", 500)

    lang = f"{LANGUAGES[language]} ({language})"
    prompt = (
        "You are marking a spoken quiz answer for a beginner.\n"
        f"Question: {q.strip()}\n"
        f"Answer (speech-to-text transcript): {answer.strip()}\n\n"
        "Judge by meaning, not exact wording. The transcript may contain small "
        "speech-to-text mistakes (misheard or misspelled words, missing "
        "punctuation); forgive those if the intended answer is clear.\n"
        f"Write feedback in {lang}. If the answer is correct, give a short "
        "confirmation. If it is wrong, state the right answer in one or two "
        "simple sentences a beginner can follow."
    )

    response, err = call_claude(
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": CHECK_SCHEMA}},
    )
    if err:
        return err

    try:
        text = next(b.text for b in response.content if b.type == "text")
        result = json.loads(text)
        return jsonify(correct=bool(result["correct"]), feedback=result["feedback"])
    except (StopIteration, ValueError, KeyError, TypeError):
        return error("Claude returned an unreadable evaluation", 502)


if __name__ == "__main__":
    app.run(debug=True)
