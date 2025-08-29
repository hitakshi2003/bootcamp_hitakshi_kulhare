import pandas as pd
import yfinance as yf

def download_data(ticker: str, start: str, end: str, auto_adjust: bool = True) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker (e.g. 'AAPL').
        start (str): Start date (YYYY-MM-DD).
        end (str): End date (YYYY-MM-DD).
        auto_adjust (bool): Adjust OHLC automatically for splits/dividends.

    Returns:
        pd.DataFrame: DataFrame with Date + OHLCV data.
    """

    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    return df.reset_index()  # gives a 'Date' column
    
    