# Bootcamp Repository

## Folder Structure

- **homework/** → All homework contributions are submitted here.
  - Each homework has its own subfolder (`homework0`, `homework1`, etc.).
  - Include all required files for grading.
- **project/** → All project contributions are submitted here.
  - Keep project files organized and clearly named.
- **class_materials/** → Local storage for class materials.  
  *Never pushed to GitHub.*
- **data/** → Storage for datasets.
  - `data/raw/` — Raw CSV files
  - `data/processed/` — Optimized Parquet files

---

## Data Storage Workflow

This project demonstrates a reproducible data storage workflow using **CSV** and **Parquet** formats, with paths managed via environment variables.

- **CSV** → human-readable, easy to share  
- **Parquet** → efficient for analytics, preserves schema & data types  

Environment variables are stored in a `.env` file and loaded in notebooks/scripts to ensure reproducibility across systems.

---

## Flask API

A minimal **Flask service** is included (`homework/homework13/app_flask.py`) to serve the trained model artifacts.

- `GET /health` → Liveness/readiness probe  
- `POST /predict` → Returns probability and binary prediction for input features  
- `GET /plot` → Returns a PNG chart of closing prices (supports ticker + date range)

**Run locally:**
```bash
conda activate bootcamp_env
cd homework/homework13
python app_flask.py
