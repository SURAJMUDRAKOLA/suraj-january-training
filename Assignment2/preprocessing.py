from __future__ import annotations

import pandas as pd


RENAME_MAP = {
    "dining_rating": "rating",
    "prices": "cost",
}

IRRELEVANT_COLUMNS = ["item_name"]


def load_dataset(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [col.strip().lower().replace(" ", "_") for col in cleaned.columns]
    return cleaned


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned.loc[cleaned[col].isin(["nan", "None", ""]), col] = pd.NA
    return cleaned


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    fixed = df.copy()
    numeric_columns = [
        "rating",
        "delivery_rating",
        "dining_votes",
        "delivery_votes",
        "votes",
        "cost",
    ]

    for col in numeric_columns:
        if col in fixed.columns:
            fixed[col] = pd.to_numeric(
                fixed[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    return fixed


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()

    for col in filled.select_dtypes(include=["int64", "float64"]).columns:
        filled[col] = filled[col].fillna(filled[col].median())

    for col in filled.select_dtypes(include="object").columns:
        if filled[col].isna().sum() == 0:
            continue

        if filled[col].isna().mean() > 0.30:
            filled[col] = filled[col].fillna("Unknown")
        else:
            mode_value = filled[col].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            filled[col] = filled[col].fillna(fill_value)

    return filled


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def cap_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    capped = df.copy()
    numeric_columns = capped.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_columns:
        q1 = capped[col].quantile(0.25)
        q3 = capped[col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        capped[col] = capped[col].clip(lower=lower_bound, upper=upper_bound)

    return capped


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[col for col in IRRELEVANT_COLUMNS if col in df.columns], errors="ignore")


def add_price_category(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    bins = [
        -float("inf"),
        enriched["cost"].quantile(0.25),
        enriched["cost"].quantile(0.50),
        enriched["cost"].quantile(0.75),
        float("inf"),
    ]
    labels = ["Low", "Medium", "High", "Very High"]
    enriched["price_category"] = pd.cut(
        enriched["cost"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)
    return enriched


def preprocess_data(file_path: str) -> pd.DataFrame:
    df = load_dataset(file_path)
    df = clean_column_names(df)
    df = df.rename(columns=RENAME_MAP)
    df = clean_text_columns(df)
    df = fix_data_types(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = cap_outliers_iqr(df)
    df = drop_irrelevant_columns(df)
    df = add_price_category(df)
    return df
