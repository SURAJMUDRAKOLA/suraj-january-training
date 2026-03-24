from __future__ import annotations

from pathlib import Path

from encoding import apply_all_encodings
from preprocessing import preprocess_data
from scaling import apply_all_scaling


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "zomato.csv"
    output_path = base_dir / "zomato_processed.csv"

    print("Loading and preprocessing dataset...")
    df = preprocess_data(str(data_path))

    print("Applying categorical encoding...")
    df = apply_all_encodings(df)

    print("Applying feature scaling...")
    df = apply_all_scaling(df, ["rating", "votes", "cost"])

    df.to_csv(output_path, index=False)
    print(f"Final processed dataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
