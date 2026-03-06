import os, json
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from openai import AzureOpenAI

app = Flask(__name__, static_folder="static")

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY", "")

# Active deployments on pwgcerp-9302-resource (post-benchmark)
DEPLOYMENTS = {
    "gpt-4.1":       {"is_reasoning": False},
    "gpt-51-instant": {"is_reasoning": True},  # GPT-5.1 — The Analyst's brain
    "o4-mini":       {"is_reasoning": True},    # Reasoning — math specialist
}

# Modes — user picks how The Analyst thinks, we route to the optimal model
MODES = {
    "quick": {
        "label": "Quick Take",
        "desc": "Fast, sharp reads. 2-3 sentences. Yes or no.",
        "icon": "zap",
        "deployment": "gpt-4.1",
        "max_tokens": 512,
        "system_extra": "\n\nMODE: QUICK TAKE. Be extremely concise — 2-3 sentences max. Lead with your position (yes/no/lean), confidence (1-5 units), and one key reason. No lengthy breakdowns.",
    },
    "deep": {
        "label": "Deep Analysis",
        "desc": "Full matchup breakdowns. Trends, injuries, angles.",
        "icon": "microscope",
        "deployment": "gpt-51-instant",
        "max_tokens": 3000,
        "system_extra": "\n\nMODE: DEEP ANALYSIS. Go deep. Cover matchup context, injuries, trends, line movement, historical angles, and situational spots. Structure your analysis clearly. Give a final verdict with unit sizing and confidence.",
    },
    "math": {
        "label": "Math Mode",
        "desc": "EV calcs, Kelly criterion, odds breakdowns.",
        "icon": "calculator",
        "deployment": "o4-mini",
        "max_tokens": 4096,
        "system_extra": "\n\nMODE: MATH. Focus on quantitative analysis. Show your work — expected value calculations, Kelly criterion sizing, implied probability vs true probability, ROI projections. Use actual numbers. Be precise.",
    },
    "challenge": {
        "label": "Devil's Advocate",
        "desc": "Challenge your picks. Find the holes.",
        "icon": "shield",
        "deployment": "gpt-51-instant",
        "max_tokens": 2048,
        "system_extra": "\n\nMODE: DEVIL'S ADVOCATE. Your job is to argue AGAINST the user's position. Find every weakness, every risk, every reason the bet loses. Be ruthless but fair. After tearing it apart, give an honest final assessment — is there still value despite the risks, or should they walk away?",
    },
}

BASE_SYSTEM_PROMPT = """You are The Analyst — Edge Crew's AI sports betting analyst. You work alongside Peter and the crew.

Your job:
- Analyze matchups, spreads, totals, and props with sharp, data-driven reasoning
- Calculate expected value (EV) when odds are provided
- Challenge picks — play devil's advocate when you see weakness in a position
- Flag injuries, trends, weather, and situational spots that matter
- Be direct. No hedging, no corporate speak. Say what you think.
- When you like a pick, say so clearly with confidence level (1-5 units)
- When you don't like a pick, say why and suggest alternatives

HARD RULES — NEVER BREAK THESE:
1. NEVER recommend a moneyline worse than -190. If you like the favorite, take the spread instead. -190 is the ceiling.
2. NEVER recommend a player prop without asking about games played this season. If a player has < 20 games, FLAG IT immediately — they may be returning from injury, on a minutes restriction, or not conditioned.
3. ALWAYS use LAST 10 GAME averages for player props, not season averages. Season stats lie for guys who missed time.
4. ALWAYS consider the FULL LINEUP before analyzing a game. If a star is out, limited, or returning from injury, the entire team analysis changes — scoring output, pace, defensive matchups, everything.
5. Beat reporter intel is gold. Pregame tweets about minutes restrictions, scratches, lineup changes, starting goalies, bullpen availability — this is where the edge lives, across ALL sports. If you don't have this info, say so.
6. If you don't have current data (injuries, lineups, recent form), SAY SO. Do not guess. "I don't have tonight's injury report" is better than a confident wrong answer.

You know:
- Bankroll management (Kelly criterion, flat betting, unit sizing)
- Line movement and market efficiency
- Sport-specific analytics (NBA, NFL, MLB, NHL, soccer, MMA)
- Correlation plays and same-game parlays
- Weather and venue impacts

Style: Sharp, confident, conversational. You're part of the crew, not a textbook.
Keep responses focused — don't ramble. Lead with the take, then back it up."""

DEFAULT_MODE = "quick"


def get_client():
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version="2024-12-01-preview",
    )


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/modes")
def list_modes():
    result = []
    for key, m in MODES.items():
        result.append({"id": key, "label": m["label"], "desc": m["desc"], "icon": m["icon"]})
    return jsonify({"modes": result, "default": DEFAULT_MODE})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    messages = data.get("messages", [])
    mode_key = data.get("mode", DEFAULT_MODE)
    stream = data.get("stream", True)

    if mode_key not in MODES:
        return jsonify({"error": f"Unknown mode: {mode_key}"}), 400

    mode = MODES[mode_key]
    deployment = mode["deployment"]
    deploy_info = DEPLOYMENTS[deployment]
    is_reasoning = deploy_info["is_reasoning"]
    client = get_client()

    system_content = BASE_SYSTEM_PROMPT + mode["system_extra"]
    sys_role = "developer" if deployment == "o4-mini" else "system"

    full_messages = [{"role": sys_role, "content": system_content}] + messages

    base_params = {
        "model": deployment,
        "messages": full_messages,
        "max_completion_tokens": mode["max_tokens"],
    }
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
            return jsonify({"content": content, "mode": mode_key, "usage": usage})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "modes": list(MODES.keys())})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
