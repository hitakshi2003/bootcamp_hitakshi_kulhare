from flask import Flask, request, jsonify, send_file
import os, io, json
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent  # repo root
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "project" / "model" / "model.joblib"))
FEATURES_PATH = Path(os.getenv("FEATURES_PATH", BASE_DIR / "project" / "model" / "feature_names.json"))
DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "project" / "data" / "raw" / "german.data-numeric"))

app = Flask(__name__)

model = None
feature_names = None

def load_model():
    global model, feature_names
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    feature_names = json.loads(FEATURES_PATH.read_text()) if FEATURES_PATH.exists() else None

load_model()

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": model is not None,
        "features_loaded": feature_names is not None
    }

def _df_from_payload(payload, feature_names):
    if payload is None:
        raise ValueError("Missing 'features' in JSON body.")
    if isinstance(payload, dict):
        row = {col: payload.get(col, 0) for col in feature_names}
        return pd.DataFrame([row], columns=feature_names)
    elif isinstance(payload, list):
        if len(payload) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} values, got {len(payload)}.")
        return pd.DataFrame([payload], columns=feature_names)
    else:
        raise ValueError("Invalid 'features' type; must be object or array.")

@app.post("/predict")
def predict():
    if model is None or feature_names is None:
        return jsonify({"error": "Model or feature names not loaded. Train & save first."}), 500
    try:
        data = request.get_json(force=True)
        X1 = _df_from_payload(data.get("features"), feature_names)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    proba1 = float(model.predict_proba(X1)[0, 1])
    pred = int(proba1 >= 0.5)
    return jsonify({"prediction": pred, "prob_default": round(proba1, 6)})

@app.get("/plot")
def plot():
    col = request.args.get("col", "credit_amount")
    bins = int(request.args.get("bins", 30))
    if not DATA_PATH.exists():
        # Demo plot — guarantees non-empty PNG even if dataset missing
        x = np.linspace(0, 2*np.pi, 200)
        y = np.sin(x)
        fig = plt.figure()
        plt.plot(x, y)
        plt.title("Demo sine wave (DATA_PATH not found)")
    else:
        df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, engine="python")
        # Try to label columns using saved feature_names (+ target as last)
        if feature_names and df.shape[1] == len(feature_names) + 1:
            df.columns = feature_names + ["target"]
        elif df.shape[1] == 25:
            df.columns = [
                "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
                "savings_status", "employment_since", "installment_rate_pct", "personal_status_sex", "other_debtors",
                "residence_since", "property_magnitude", "age_years", "other_installment_plans", "housing",
                "number_credits", "job", "people_liable", "telephone", "foreign_worker",
                "existing_credit", "dependents", "own_telephone", "foreign_worker_flag", "target"
            ]
        # else: leave numeric indexing
        if col not in df.columns:
            return jsonify({"error": f"Column '{col}' not found. Choose from: {', '.join(map(str, df.columns))}"}), 400
        fig = plt.figure()
        plt.hist(df[col].values, bins=bins)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col); plt.ylabel("Count")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png", download_name="plot.png")

if __name__ == "__main__":
    app.run(debug=True)