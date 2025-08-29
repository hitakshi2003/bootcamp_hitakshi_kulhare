# app_task.py (replace my_task with this)
import argparse, json, logging, sys
from datetime import datetime
from pathlib import Path
import pandas as pd

REQUIRED_COLS = ["Date", "Close", "Volume"]

def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".csv"}:
        return pd.read_csv(p)
    return pd.read_json(p)  # default: json

def _write_json(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(df.to_json(orient="records", date_format="iso", indent=2))

def my_task(input_path: str, output_path: str) -> None:
    """Clean step: load → validate schema → dedupe/sort → drop NAs → write JSON."""
    logging.info("[clean] start")
    df = _read_any(input_path)

    # basic schema check
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")

    # ensure datetime + sort
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # dedupe & drop nans on key fields
    before = len(df)
    df = df.drop_duplicates(subset=["Date"]).dropna(subset=["Close", "Volume"])
    logging.info("[clean] rows: %d → %d after dedupe/NA drop", before, len(df))

    # optional: clip extreme Close values (robust cap)
    q_low, q_hi = df["Close"].quantile([0.001, 0.999])
    df["Close"] = df["Close"].clip(lower=q_low, upper=q_hi)

    # metadata
    df.attrs["run_at"] = datetime.utcnow().isoformat()

    _write_json(df[REQUIRED_COLS], output_path)
    logging.info("[clean] wrote %s", output_path)

def _ensure_sample_input(path: str) -> None:
    """Create a tiny sample JSON file if it doesn't exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        sample = [
            {"Date": "2025-01-02", "Close": 100.0, "Volume": 12000},
            {"Date": "2025-01-03", "Close": 101.5, "Volume": 15000},
            {"Date": "2025-01-06", "Close":  99.8, "Volume": 13500},
        ]
        p.write_text(json.dumps(sample, indent=2))

def main(argv=None):
    parser = argparse.ArgumentParser(description="Homework task: clean")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

    # Create a sample input if missing (useful in notebooks or first runs)
    if not Path(args.input).exists():
        logging.warning("[clean] '%s' not found; creating a sample file.", args.input)
        _ensure_sample_input(args.input)

    my_task(args.input, args.output)

if __name__ == "__main__":
    # Detect Jupyter vs Terminal run
    in_notebook = any("ipykernel" in m for m in sys.modules)

    if in_notebook:
        # Safe defaults for notebook users
        demo_in = "data/prices_raw.json"
        demo_out = "data/prices_clean.json"
        logging.getLogger().setLevel(logging.INFO)
        print(f"[demo] Running in notebook with --input {demo_in} --output {demo_out}")
        main(["--input", demo_in, "--output", demo_out])
    else:
        # Real CLI: uses sys.argv (e.g., python app_task.py --input ... --output ...)
        main()