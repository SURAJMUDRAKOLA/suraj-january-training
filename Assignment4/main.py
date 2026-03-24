import os

from svm_model import (
    load_and_clean_data,
    save_class_distribution_plot,
    save_cleaned_data,
    save_confusion_matrix_plot,
    save_predictions,
    save_results_summary,
    train_svm_classifier,
)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "Spam Email Detection.xlsx")

    cleaned_data_path = os.path.join(base_dir, "spam_cleaned.csv")
    predictions_path = os.path.join(base_dir, "spam_predictions.csv")
    summary_path = os.path.join(base_dir, "results_summary.txt")
    class_plot_path = os.path.join(base_dir, "class_distribution.png")
    confusion_plot_path = os.path.join(base_dir, "confusion_matrix.png")

    print("Loading dataset...")
    df = load_and_clean_data(dataset_path)
    print(f"Cleaned dataset shape: {df.shape}")

    save_cleaned_data(df, cleaned_data_path)
    save_class_distribution_plot(df, class_plot_path)

    print("Training SVM model...")
    _, _, metrics, predictions = train_svm_classifier(df)

    save_predictions(predictions, predictions_path)
    save_confusion_matrix_plot(metrics["confusion_matrix"], confusion_plot_path)
    save_results_summary(df, metrics, summary_path)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    print("\nFiles saved:")
    print(f"Cleaned dataset: {cleaned_data_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Results summary: {summary_path}")
    print(f"Class distribution plot: {class_plot_path}")
    print(f"Confusion matrix plot: {confusion_plot_path}")


if __name__ == "__main__":
    main()
