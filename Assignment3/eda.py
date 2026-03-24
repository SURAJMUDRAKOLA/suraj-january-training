from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from linear_regression import TARGET_COLUMN, clean_data, load_dataset


def save_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, fontsize=7)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return correlation_matrix


def save_scatter_plots(df: pd.DataFrame, output_dir: Path) -> list[str]:
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    top_features = (
        numeric_df.corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .abs()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    for feature in top_features:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[feature], df[TARGET_COLUMN], alpha=0.4)
        plt.xlabel(feature)
        plt.ylabel(TARGET_COLUMN)
        plt.title(f"{feature} vs {TARGET_COLUMN}")
        plt.tight_layout()
        plt.savefig(output_dir / f"scatter_{feature}_vs_{TARGET_COLUMN}.png", dpi=200)
        plt.close()

    return top_features


def identify_multicollinearity(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.75,
) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    columns = correlation_matrix.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            feature_1 = columns[i]
            feature_2 = columns[j]
            corr_value = abs(correlation_matrix.iloc[i, j])

            if feature_1 == TARGET_COLUMN or feature_2 == TARGET_COLUMN:
                continue

            if corr_value >= threshold:
                pairs.append((feature_1, feature_2, float(corr_value)))

    return pairs


def run_eda(file_path: str, output_dir: str) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_dataset(file_path)
    df = clean_data(df)

    correlation_matrix = save_correlation_heatmap(df, output_path / "correlation_heatmap.png")
    top_features = save_scatter_plots(df, output_path)
    multicollinearity_pairs = identify_multicollinearity(correlation_matrix)

    return {
        "top_features": top_features,
        "multicollinearity_pairs": multicollinearity_pairs,
    }
