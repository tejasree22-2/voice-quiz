import array
import json
import math
import os
import subprocess

import anthropic
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
# Saarika is being retired; Sarvam's replacement is saaras:v3 in "transcribe" mode.
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL") or "saaras:v3"
MAX_AUDIO_SECONDS = 30  # Sarvam REST limit
SILENCE_RMS = 100  # of 32768; below this the clip is treated as silence

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


def decode_audio(raw):
    """Convert any ffmpeg-readable audio (e.g. browser WebM/Opus) to 16 kHz mono
    16-bit PCM. Returns (samples, None) or (None, error message)."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
             "-f", "s16le", "pipe:1"],
            input=raw, capture_output=True, timeout=30,
        )
    except FileNotFoundError:
        return None, "ffmpeg is not installed on the server"
    except subprocess.TimeoutExpired:
        return None, "Audio conversion timed out"
    if proc.returncode != 0 or not proc.stdout:
        return None, "Could not read the audio file (unsupported or corrupt)"
    samples = array.array("h")
    samples.frombytes(proc.stdout[: len(proc.stdout) // 2 * 2])
    return samples, None


def to_wav(samples):
    """Wrap 16 kHz mono PCM samples in a WAV container."""
    data = samples.tobytes()
    header = (
        b"RIFF" + (36 + len(data)).to_bytes(4, "little") + b"WAVEfmt "
        + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little") + (16000).to_bytes(4, "little")
        + (32000).to_bytes(4, "little") + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little") + b"data" + len(data).to_bytes(4, "little")
    )
    return header + data


@app.route("/transcribe", methods=["POST"])
def transcribe():
    upload = request.files.get("audio") or request.files.get("file")
    language = request.form.get("language")

    if upload is None:
        return error("Send the recording as multipart field 'audio'", 400)
    if language not in LANGUAGES:
        return error("'language' must be one of: " + ", ".join(LANGUAGES), 400)
    if not SARVAM_API_KEY:
        return error("SARVAM_API_KEY is not configured on the server", 500)

    raw = upload.read()
    if not raw:
        return error("The audio file is empty", 400)

    # Browsers record WebM/Opus. Sarvam lists WebM as supported, but MediaRecorder
    # files have no duration header; normalising to WAV is more reliable and lets
    # us reject silence and over-long clips before spending an API call.
    samples, problem = decode_audio(raw)
    if problem:
        return error(problem, 500 if "ffmpeg" in problem else 400)
    if len(samples) == 0:
        return error("The audio contains no sound", 400)
    if len(samples) / 16000 > MAX_AUDIO_SECONDS:
        return error(f"Audio is longer than {MAX_AUDIO_SECONDS} seconds", 413)
    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    if rms < SILENCE_RMS:
        return error("The audio is silent - nothing was heard, please try again", 422)

    try:
        resp = requests.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("audio.wav", to_wav(samples), "audio/wav")},
            data={"model": SARVAM_STT_MODEL, "mode": "transcribe",
                  "language_code": language},
            timeout=60,
        )
    except requests.Timeout:
        return error("Sarvam speech-to-text timed out", 504)
    except requests.RequestException:
        return error("Could not reach Sarvam speech-to-text", 502)

    if resp.status_code in (401, 403):
        return error("Sarvam rejected the API key", 502)
    if resp.status_code == 429:
        return error("Sarvam rate limit reached, try again shortly", 429)
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message") or resp.text
        except ValueError:
            detail = resp.text
        return error(f"Sarvam error ({resp.status_code}): {detail[:200]}", 502)

    text = (resp.json().get("transcript") or "").strip()
    if not text:
        return error("No speech was recognised in the audio", 422)
    return jsonify(text=text)


if __name__ == "__main__":
    app.run(debug=True)
