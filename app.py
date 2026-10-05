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

    try:
        response = claude.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        return error("Invalid Anthropic API key", 502)
    except anthropic.RateLimitError:
        return error("Claude API rate limit reached, try again shortly", 429)
    except anthropic.APITimeoutError:
        return error("Claude API timed out", 504)
    except anthropic.APIConnectionError:
        return error("Could not reach the Claude API", 502)
    except anthropic.APIStatusError as e:
        return error(f"Claude API error: {e.message}", 502)

    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    if not text:
        return error("Claude returned no question", 502)
    return jsonify(question=text)


if __name__ == "__main__":
    app.run(debug=True)
