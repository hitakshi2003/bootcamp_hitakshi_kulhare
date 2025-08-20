#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# src/cleaning.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fill_missing_median(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Fill missing values in specified columns with the median of each column.
    """
    for col in columns:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
    return df

def drop_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop columns with more than 'threshold' proportion of missing values.
    """
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    df = df.drop(columns=cols_to_drop)
    return df

def normalize_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Normalize specified columns using Min-Max Scaling.
    """
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


# In[ ]:




