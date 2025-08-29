from pathlib import Path; import joblib, json
def save_artifacts(path, model, imputer, scaler, meta):
    p = Path(path); p.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p/"model.pkl")
    joblib.dump(imputer, p/"imputer.pkl")
    joblib.dump(scaler, p/"scaler.pkl")
    with open(p/"meta.json","w") as f: json.dump(meta, f, indent=2)
def load_artifacts(path):
    p = Path(path)
    return (joblib.load(p/"model.pkl"),
            joblib.load(p/"imputer.pkl"),
            joblib.load(p/"scaler.pkl"),
            json.loads((p/"meta.json").read_text()))