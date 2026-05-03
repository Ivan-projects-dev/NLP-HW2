import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# --- LOAD DATA ---
def load_data(file):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    df["text"] = df["text"].fillna("")
    return df["text"], df["label"]


# --- EVALUATION ---
def evaluate_model(name, model, X_train, y_train, X_dev, y_dev, X_test, y_test):
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)

    # DEV
    y_dev_pred = model.predict(X_dev)
    print("DEV Accuracy:", accuracy_score(y_dev, y_dev_pred))
    print("DEV Macro F1:", f1_score(y_dev, y_dev_pred, average="macro"))

    # TEST
    y_test_pred = model.predict(X_test)
    print("TEST Accuracy:", accuracy_score(y_test, y_test_pred))
    print("TEST Macro F1:", f1_score(y_test, y_test_pred, average="macro"))

    return y_test_pred


if __name__ == "__main__":
    # --- DATA ---
    X_train, y_train = load_data("train_tokenized.tsv")
    X_dev, y_dev = load_data("dev_tokenized.tsv")
    X_test, y_test = load_data("test_tokenized.tsv")

    # --- TF-IDF MODEL ---
    tfidf_model = Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=10000)),
        ("classifier", LogisticRegression(
            solver="saga",
            max_iter=200,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    # --- COUNT MODEL (with n-grams) ---
    count_model = Pipeline([
        ("vectorizer", CountVectorizer(
            ngram_range=(1,2),
            max_features=8000
        )),
        ("classifier", LogisticRegression(
            solver="saga",
            max_iter=200,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    # --- RUN BOTH MODELS ---
    y_pred_tfidf = evaluate_model(
        "TF-IDF",
        tfidf_model,
        X_train, y_train,
        X_dev, y_dev,
        X_test, y_test
    )

    y_pred_count = evaluate_model(
        "CountVectorizer",
        count_model,
        X_train, y_train,
        X_dev, y_dev,
        X_test, y_test
    )

    # --- CONFUSION MATRIX (TF-IDF) ---
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred_tfidf, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues",
                xticklabels=labels,
                yticklabels=labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (TF-IDF)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig("confusion_matrix.png")
    print("\nConfusion matrix saved to file")