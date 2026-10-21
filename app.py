"""
app.py – Flask Web App Nhận Diện Chữ Số Viết Tay (MNIST)
=========================================================
Pipeline:
  Người dùng vẽ chữ số trên canvas HTML
    → gửi ảnh dưới dạng base64 PNG
    → Flask tiền xử lý (grayscale, invert, resize 28×28, normalize)
    → CNN predict (10 xác suất)
    → trả JSON về frontend
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# ─────────────────────────── Paths ────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
MODEL_PATH    = BASE_DIR / "artifacts" / "cnn_mnist_model.keras"
METADATA_PATH = BASE_DIR / "artifacts" / "model_metadata.json"

# ─────────────────────────── Flask app ────────────────────────────
app = Flask(__name__)

# ─────────────────────────── Model loading ────────────────────────
_model    = None
_metadata: dict = {}


def _load_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_model():
    """
    Lazy-load mô hình Keras.
    Dùng biến module-level để chỉ load 1 lần (thread-safe với Gunicorn --preload).
    """
    global _model
    if _model is None and MODEL_PATH.exists():
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        from tensorflow import keras  # noqa: PLC0415

        try:
            _model = keras.models.load_model(str(MODEL_PATH))
            print(f"[INFO] Model loaded successfully: {MODEL_PATH.name}")
        except Exception as exc:
            print(f"[ERROR] Model load failed: {exc}")
    return _model


# Load metadata ngay khi khởi động (nhẹ)
_metadata = _load_metadata()


# ─────────────────────────── Image preprocessing ──────────────────

def preprocess_canvas_image(data_url: str) -> np.ndarray:
    """
    Chuyển đổi ảnh canvas HTML base64 → numpy array chuẩn MNIST.

    Pipeline biến đổi:
    ┌─────────────────────────────────────────────────────┐
    │  Canvas PNG (RGBA, ~280×280, nền trắng, chữ đen)  │
    │       ↓ decode base64                               │
    │  PIL RGBA image                                     │
    │       ↓ blend lên nền trắng → convert 'L'          │
    │  Grayscale image (H×W, 0–255)                       │
    │       ↓ LANCZOS resize                              │
    │  28×28 grayscale                                    │
    │       ↓ arr = 255 − arr  (invert)                   │
    │  MNIST style: nền đen=0, chữ sáng                   │
    │       ↓ / 255.0                                     │
    │  float32 [0.0, 1.0]                                 │
    │       ↓ reshape                                     │
    │  (1, 28, 28, 1)  ← sẵn sàng cho CNN.predict()     │
    └─────────────────────────────────────────────────────┘

    Args:
        data_url: Data URL từ canvas.toDataURL('image/png')
                  dạng "data:image/png;base64,<base64_string>"

    Returns:
        numpy array shape (1, 28, 28, 1) dtype float32
    """
    # 1. Decode base64
    _, b64 = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64)

    # 2. Mở ảnh; blend RGBA lên nền trắng → tránh viền đen khi convert
    raw = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    bg  = Image.new("RGBA", raw.size, (255, 255, 255, 255))
    bg.paste(raw, mask=raw.split()[3])          # dùng alpha làm mask
    gray = bg.convert("L")                      # → grayscale

    # 3. Resize về 28×28
    gray = gray.resize((28, 28), Image.LANCZOS)

    # 4. Numpy + invert + normalize
    arr = np.array(gray, dtype=np.float32)
    arr = 255.0 - arr                           # invert: canvas trắng → MNIST đen
    arr = arr / 255.0                           # normalize [0, 1]

    # 5. Reshape cho CNN
    arr = arr.reshape(1, 28, 28, 1)
    return arr


# ─────────────────────────── Routes ───────────────────────────────

@app.route("/")
def index():
    """Trang chủ - hiển thị canvas vẽ và phần kết quả."""
    return render_template(
        "index.html",
        model_loaded=MODEL_PATH.exists(),
        metadata=_metadata,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint: nhận ảnh canvas → trả về dự đoán CNN.

    Request JSON:
        { "image": "data:image/png;base64,..." }

    Response JSON (200):
        {
          "digit"        : 5,
          "confidence"   : 98.72,
          "probabilities": [0.01, 0.00, ..., 98.72, ...],   // 10 phần tử (%)
          "top3"         : [
            {"digit": 5, "prob": 98.72},
            {"digit": 6, "prob":  0.87},
            {"digit": 3, "prob":  0.21}
          ]
        }

    Response JSON (4xx / 5xx):
        { "error": "<mô tả lỗi>" }
    """
    model = get_model()
    if model is None:
        return jsonify(
            error=(
                "Mô hình chưa sẵn sàng. "
                "Hãy chạy notebook `cnn_mnist_training.ipynb` để train và lưu mô hình trước!"
            )
        ), 503

    try:
        payload    = request.get_json(force=True, silent=True) or {}
        image_data = payload.get("image", "")

        if not image_data:
            return jsonify(error="Không nhận được dữ liệu ảnh."), 400

        # Tiền xử lý
        arr = preprocess_canvas_image(image_data)

        # Dự đoán
        proba     = model.predict(arr, verbose=0)[0]          # shape (10,)
        predicted = int(np.argmax(proba))
        conf      = float(proba[predicted]) * 100

        # Top-3
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [
            {"digit": int(i), "prob": round(float(proba[i]) * 100, 2)}
            for i in top3_idx
        ]

        return jsonify(
            digit=predicted,
            confidence=round(conf, 2),
            probabilities=[round(float(p) * 100, 2) for p in proba],
            top3=top3,
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify(error=f"Lỗi xử lý: {exc}"), 500


@app.route("/health")
def health():
    """Health-check endpoint – Render dùng để kiểm tra app còn sống."""
    return jsonify(
        status="ok",
        model_ready=MODEL_PATH.exists(),
        accuracy=_metadata.get("accuracy"),
    )


# ─────────────────────────── Entry point ──────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
