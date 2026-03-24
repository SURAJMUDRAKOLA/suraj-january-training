from __future__ import annotations

from pathlib import Path

from eda import run_eda
from linear_regression import (
    clean_data,
    evaluate_model,
    get_coefficients,
    load_dataset,
    split_data,
    train_model,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "dataset.csv"

    print("Running EDA...")
    eda_summary = run_eda(str(dataset_path), str(base_dir))

    print("Loading and cleaning data...")
    df = load_dataset(str(dataset_path))
    df = clean_data(df)

    print("Splitting data...")
    x_train, x_test, y_train, y_test = split_data(df)

    print("Training Linear Regression model...")
    model, scaler = train_model(x_train, y_train)

    print("Evaluating model...")
    _, mse, r2 = evaluate_model(model, scaler, x_test, y_test)
    coefficients = get_coefficients(model, x_train.columns.tolist())

    positive_features = coefficients[coefficients["coefficient"] > 0].head(5)
    negative_features = coefficients[coefficients["coefficient"] < 0].head(5)

    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Top EDA features related to price: {eda_summary['top_features']}")
    print(f"High-correlation pairs found: {len(eda_summary['multicollinearity_pairs'])}")

    print("\nTop features that increase price:")
    print(positive_features[["feature", "coefficient"]].to_string(index=False))

    print("\nTop features that decrease price:")
    print(negative_features[["feature", "coefficient"]].to_string(index=False))


if __name__ == "__main__":
    main()
