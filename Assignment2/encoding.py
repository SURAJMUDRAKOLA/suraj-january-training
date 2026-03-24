from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, dtype=int)


def label_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    encoded = df.copy()
    for col in columns:
        encoder = LabelEncoder()
        encoded[f"{col}_label"] = encoder.fit_transform(encoded[col].astype(str))
    return encoded


def ordinal_encode(df: pd.DataFrame, column: str, categories: list[str]) -> pd.DataFrame:
    encoded = df.copy()
    encoder = OrdinalEncoder(categories=[categories], dtype=float)
    encoded[f"{column}_ordinal"] = encoder.fit_transform(encoded[[column]].astype(str))
    return encoded


def frequency_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    encoded = df.copy()
    for col in columns:
        frequency_map = encoded[col].value_counts().to_dict()
        encoded[f"{col}_frequency"] = encoded[col].map(frequency_map)
    return encoded


def target_encode(df: pd.DataFrame, column: str, target_column: str) -> pd.DataFrame:
    encoded = df.copy()
    target_map = encoded.groupby(column)[target_column].mean().to_dict()
    encoded[f"{column}_target"] = encoded[column].map(target_map)
    return encoded


def apply_all_encodings(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    encoded = one_hot_encode(encoded, ["city"])
    encoded = label_encode(encoded, ["best_seller"])
    encoded = ordinal_encode(encoded, "price_category", ["Low", "Medium", "High", "Very High"])
    encoded = frequency_encode(encoded, ["cuisine", "place_name"])
    encoded = target_encode(encoded, "restaurant_name", "rating")

    columns_to_drop = [
        "restaurant_name",
        "cuisine",
        "place_name",
        "best_seller",
        "price_category",
    ]
    encoded = encoded.drop(columns=[col for col in columns_to_drop if col in encoded.columns])
    return encoded
