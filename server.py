import base64
import json
import os

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()

app = Flask(__name__, static_folder="public")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

PROMPT = """Analyze this food image and respond with a JSON object (no markdown fences):
{
  "foods": [
    {
      "name": "exact food name",
      "category": "Fruit | Vegetable | Meat | Seafood | Grain | Dairy | Dessert | Beverage | Snack | Dish",
      "description": "1-2 sentence description",
      "cuisine": "cuisine origin or null",
      "calories_per_serving": "~250 kcal or null",
      "dominant_colors": ["golden-brown", "red"],
      "key_ingredients": ["ingredient1", "ingredient2", "ingredient3"]
    }
  ],
  "summary": "brief description of the image",
  "confidence": "high | medium | low",
  "grid": [
    ["cat","cat","cat","cat"],
    ["cat","cat","cat","cat"],
    ["cat","cat","cat","cat"],
    ["cat","cat","cat","cat"]
  ],
  "category_pct": {"noodles":0,"rice":0,"meat":0,"bread":0,"fish":0,"vegetables":0,"other":0}
}

grid: 4 rows × 4 columns (row 1 = top of image, col 1 = left). Each cell is the dominant food type
in that region. Use only: noodles, rice, meat, bread, fish, vegetables, other.
category_pct: integer percentage (0-100) of visible area for each category. Values must sum to 100.
dominant_colors: 2-4 color names, e.g. "golden-brown", "red", "green", "white", "orange", "cream", "dark-brown".
key_ingredients: 3-7 main ingredients, e.g. "flour", "egg", "tomato", "cheese", "chicken", "rice", "garlic".
If no food: { "foods": [], "summary": "No food detected.", "confidence": "high",
  "grid": [["other","other","other","other"],["other","other","other","other"],
            ["other","other","other","other"],["other","other","other","other"]],
  "category_pct": {"noodles":0,"rice":0,"meat":0,"bread":0,"fish":0,"vegetables":0,"other":100} }."""


def validate_image(file):
    """Return (b64, media_type) or raise ValueError."""
    if file.content_type not in ALLOWED_TYPES:
        raise ValueError("Unsupported file type. Use JPEG, PNG, GIF, or WebP.")
    data = file.read()
    if len(data) > MAX_BYTES:
        raise ValueError("File too large. Maximum size is 10 MB.")
    return base64.standard_b64encode(data).decode(), file.content_type


def call_claude(b64: str, media_type: str) -> str:
    """Send image to Claude and return the raw text response."""
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    ) as stream:
        message = stream.get_final_message()

    for block in message.content:
        if block.type == "text":
            return block.text.strip()
    return ""


def parse_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON; fall back to raw on failure."""
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw": cleaned}


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.post("/identify")
def identify():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    try:
        b64, media_type = validate_image(request.files["image"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        raw = call_claude(b64, media_type)
    except anthropic.APIError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(parse_response(raw))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"Food Identifier running at:")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
