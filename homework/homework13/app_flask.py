import sys
from pathlib import Path

_here = Path(__file__).resolve()
for p in [_here.parent] + list(_here.parents):
    if (p / "src" / "__init__.py").exists():
        sys.path.insert(0, str(p))  
        repo_root = p
        break
else:
    raise RuntimeError(
        "Could not find 'src/__init__.py'. "
        "Run from your project root or ensure a 'src/' package exists."
    )
# --------------------------------------------------------------------

from flask import Flask, request, jsonify, Response
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt
import io

from src.io import load_artifacts 

ART_DIR = repo_root / "artifacts"

try:
    MODEL, IMPUTER, SCALER, META = load_artifacts(ART_DIR)
except Exception as e:
    MODEL = IMPUTER = SCALER = META = None
    print(f"[WARN] Could not load artifacts from {ART_DIR}: {e}")

app = Flask(__name__)

@app.get("/health")
def health():
    status = "ok" if MODEL is not None else "degraded"
    return jsonify({"status": status})

@app.post("/predict")
def predict():
    if MODEL is None or IMPUTER is None:
        return jsonify({"error": "Model not loaded. Train/export artifacts first."}), 503

    data = request.get_json(silent=True) or {}
    features = data.get("features")

    if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
        return jsonify({"error": "Body must be JSON with numeric list 'features'."}), 400

    X = np.array(features, dtype=float).reshape(1, -1)
    X_imp = IMPUTER.transform(X)
    

    if hasattr(MODEL, "predict_proba"):
        proba_up = float(MODEL.predict_proba(X_imp)[:, 1][0])
        threshold = (META or {}).get("threshold", 0.5)
        pred_up = int(proba_up >= threshold)
        return jsonify({"proba_up": round(proba_up, 6), "pred_up": pred_up})


    pred = float(MODEL.predict(X_imp)[0])
    return jsonify({"prediction": pred})

@app.get("/plot")
def plot():
    """
    Plot actual closing prices for a ticker between start/end (YYYY-MM-DD).
    Query params (all optional):
      - ticker= AAPL (default)
      - start= 2020-01-01 (default)
      - end=   today (default via yfinance)
    """
    from flask import request
    from src import download_data  # uses your src/data.py

    ticker = request.args.get("ticker", "AAPL").upper()
    start  = request.args.get("start", "2020-01-01")
    end    = request.args.get("end", None)  # let yfinance default to today if None

    # Download OHLCV
    try:
        df = download_data(ticker, start=start, end=end, auto_adjust=True)
    except Exception as e:
        return jsonify({"error": f"Failed to download data for {ticker}: {e}"}), 502

    # Basic validation
    if df is None or df.empty or "Close" not in df.columns:
        return jsonify({"error": f"No data available for {ticker} in given range."}), 404

    # Ensure plotting on Date vs Close
    # yfinance reset_index() gives 'Date' as a column already
    # Convert Date to pandas datetime in case it's not
    try:
        import pandas as pd
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        pass

    # Build figure
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Date"], df["Close"], label=f"{ticker} Close")
    ax.set_title(f"{ticker} Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)