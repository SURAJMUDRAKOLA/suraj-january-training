from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler


def min_max_scale(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaled = df.copy()
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(scaled[columns])
    for index, col in enumerate(columns):
        scaled[f"{col}_minmax"] = scaled_values[:, index]
    return scaled


def max_abs_scale(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaled = df.copy()
    scaler = MaxAbsScaler()
    scaled_values = scaler.fit_transform(scaled[columns])
    for index, col in enumerate(columns):
        scaled[f"{col}_maxabs"] = scaled_values[:, index]
    return scaled


def normalize_data(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaled = df.copy()
    scaler = Normalizer()
    scaled_values = scaler.fit_transform(scaled[columns])
    for index, col in enumerate(columns):
        scaled[f"{col}_normalized"] = scaled_values[:, index]
    return scaled


def standardize_data(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaled = df.copy()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(scaled[columns])
    for index, col in enumerate(columns):
        scaled[f"{col}_standardized"] = scaled_values[:, index]
    return scaled


def apply_all_scaling(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaled = df.copy()
    scaled = min_max_scale(scaled, columns)
    scaled = max_abs_scale(scaled, columns)
    scaled = normalize_data(scaled, columns)
    scaled = standardize_data(scaled, columns)
    return scaled
