from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "price"


def load_dataset(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [col.strip().lower() for col in cleaned.columns]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        cleaned["sale_year"] = cleaned["date"].dt.year
        cleaned["sale_month"] = cleaned["date"].dt.month
        cleaned = cleaned.drop(columns=["date"])

    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            mode_value = cleaned[col].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)
        else:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    if "id" in cleaned.columns:
        cleaned = cleaned.drop(columns=["id"])

    categorical_columns = cleaned.select_dtypes(include="object").columns.tolist()
    if categorical_columns:
        cleaned = pd.get_dummies(cleaned, columns=categorical_columns, drop_first=True)

    return cleaned


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(x, y, test_size=0.20, random_state=42)


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[LinearRegression, StandardScaler]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    return model, scaler


def evaluate_model(
    model: LinearRegression,
    scaler: StandardScaler,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.Series, float, float]:
    x_test_scaled = scaler.transform(x_test)
    predictions = pd.Series(model.predict(x_test_scaled), index=y_test.index)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return predictions, float(mse), float(r2)


def get_coefficients(model: LinearRegression, feature_names: list[str]) -> pd.DataFrame:
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_,
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    return coefficients.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
