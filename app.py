import os, json
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from openai import AzureOpenAI

app = Flask(__name__, static_folder="static")

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY", "")

# All available models on pwgcerp-9302-resource
MODELS = {
    "gpt-5.2":    {"deployment": "52-instant",    "label": "GPT-5.2",        "desc": "Flagship — deepest analysis"},
    "gpt-5.1":    {"deployment": "gpt-51-instant", "label": "GPT-5.1",       "desc": "Strong all-rounder"},
    "gpt-5-mini": {"deployment": "gpt-5-mini",    "label": "GPT-5 Mini",     "desc": "Fast + smart"},
    "gpt-4.1":    {"deployment": "gpt-4.1",       "label": "GPT-4.1",        "desc": "Reliable workhorse"},
    "o4-mini":    {"deployment": "o4-mini",        "label": "o4-mini",        "desc": "Reasoning — odds & math"},
    "gpt-4o":     {"deployment": "gpt-4o",         "label": "GPT-4o",         "desc": "Baseline comparison"},
}

DEFAULT_MODEL = "gpt-5-mini"

SYSTEM_PROMPT = """You are The Analyst — Edge Crew's AI sports betting analyst. You work alongside Peter and the crew.

Your job:
- Analyze matchups, spreads, totals, and props with sharp, data-driven reasoning
- Calculate expected value (EV) when odds are provided
- Challenge picks — play devil's advocate when you see weakness in a position
- Flag injuries, trends, weather, and situational spots that matter
- Be direct. No hedging, no corporate speak. Say what you think.
- When you like a pick, say so clearly with confidence level (1-5 units)
- When you don't like a pick, say why and suggest alternatives

You know:
- Bankroll management (Kelly criterion, flat betting, unit sizing)
- Line movement and market efficiency
- Sport-specific analytics (NBA, NFL, MLB, NHL, soccer, MMA)
- Correlation plays and same-game parlays
- Weather and venue impacts

Style: Sharp, confident, conversational. You're part of the crew, not a textbook.
Keep responses focused — don't ramble. Lead with the take, then back it up."""


def get_client():
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version="2024-12-01-preview",
    )


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/models")
def list_models():
    result = []
    for key, m in MODELS.items():
        result.append({"id": key, "label": m["label"], "desc": m["desc"]})
    return jsonify({"models": result, "default": DEFAULT_MODEL})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    messages = data.get("messages", [])
    model_key = data.get("model", DEFAULT_MODEL)
    stream = data.get("stream", True)

    if model_key not in MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    deployment = MODELS[model_key]["deployment"]
    is_reasoning = model_key in ("o4-mini", "gpt-5-mini", "gpt-5.2", "gpt-5.1")
    client = get_client()

    if is_reasoning:
        full_messages = [{"role": "developer", "content": SYSTEM_PROMPT}] + messages
    else:
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    base_params = {"model": deployment, "messages": full_messages, "max_completion_tokens": 2048}
    if not is_reasoning:
        base_params["temperature"] = 0.7

    if stream:
        def generate():
            try:
                response = client.chat.completions.create(**base_params, stream=True)
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        try:
            response = client.chat.completions.create(**base_params)
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            return jsonify({"content": content, "model": model_key, "usage": usage})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models": list(MODELS.keys())})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
