import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def load_and_clean_data(file_path):
    df = pd.read_excel(file_path)

    df = df.iloc[:, :2].copy()
    df.columns = ["label", "message"]

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df.dropna(subset=["label", "message"])
    df = df[df["message"] != ""]
    df = df.drop_duplicates().reset_index(drop=True)

    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    return df


def train_svm_classifier(df):
    x = df["message"]
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = LinearSVC(random_state=42)
    model.fit(x_train_tfidf, y_train)

    y_pred = model.predict(x_test_tfidf)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=["ham", "spam"]),
    }

    predictions = pd.DataFrame(
        {
            "message": x_test.reset_index(drop=True),
            "actual_label": y_test.map({0: "ham", 1: "spam"}).reset_index(drop=True),
            "predicted_label": pd.Series(y_pred).map({0: "ham", 1: "spam"}),
        }
    )

    return model, vectorizer, metrics, predictions


def save_class_distribution_plot(df, output_path):
    counts = df["label"].value_counts()

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color=["steelblue", "tomato"])
    plt.title("Spam vs Ham Distribution")
    plt.xlabel("Email Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_matrix_plot(confusion, output_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(confusion, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Ham", "Spam"])
    plt.yticks([0, 1], ["Ham", "Spam"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, confusion[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_results_summary(df, metrics, output_path):
    class_counts = df["label"].value_counts()

    lines = [
        "Assignment 4 - Spam Email Detection using SVM",
        "",
        f"Dataset rows after cleaning: {len(df)}",
        f"Ham messages: {class_counts.get('ham', 0)}",
        f"Spam messages: {class_counts.get('spam', 0)}",
        "",
        "Model Evaluation",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1 Score: {metrics['f1_score']:.4f}",
        "",
        "Confusion Matrix",
        str(metrics["confusion_matrix"]),
        "",
        "Classification Report",
        metrics["classification_report"],
    ]

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)


def save_predictions(predictions, output_path):
    predictions.to_csv(output_path, index=False)
