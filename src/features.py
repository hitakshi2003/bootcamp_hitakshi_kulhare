# project/src/features.py
import pandas as pd
import numpy as np

# Columns the model expects
FEATURE_COLUMNS = ["MA5", "MA10", "MA20", "Volatility20", "Lag1", "Lag2", "Volume"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Return, Return_next, Target and core technical indicators.

    Assumes df has at least: ['Close', 'Volume'].
    Works with a single-ticker OHLCV dataframe downloaded via yfinance (auto_adjust=True is fine).
    """
    df = df.copy()

    # Basic returns & label
    df["Return"] = df["Close"].pct_change()
    df["Return_next"] = df["Return"].shift(-1)
    df["Target"] = (df["Return_next"] > 0).astype(int)

    # Technical indicators
    df["MA5"]  = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Volatility20"] = df["Return"].rolling(20).std()
    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)

    # Replace infinities so imputer can handle them later
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df