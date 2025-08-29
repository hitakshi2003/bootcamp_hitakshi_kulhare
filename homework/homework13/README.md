# Financial Forecasting API

This project provides a simple **Flask API** for:
- Serving a trained ML model (RandomForest) to predict stock return direction.
- Returning probability predictions via `/predict`.
- Displaying quick diagnostic plots via `/plot`.

## Features
- `POST /predict` → accepts JSON `{"features": [ ... ]}` and returns probabilities/prediction.
- `GET /plot` → returns a PNG chart of stock closing price (supports ticker + date range).
- `GET /health` → quick health check (reports if artifacts are loaded).



