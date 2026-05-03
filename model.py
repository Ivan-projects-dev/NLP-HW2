import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, f1_score)

def load_data(file):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    df["text"] = df["text"].fillna("")
    return df["text"], df["label"]

if __name__ == "__main__":
    X_train, y_train = load_data("train_tokenized.tsv")
    X_dev, y_dev = load_data("dev_tokenized.tsv")
    X_test, y_test = load_data("test_tokenized.tsv")

    model = Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=15000)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=300))
    ])
    model.fit(X_train, y_train)

    y_pred_dev = model.predict(X_dev)
    print("\n=== DEV RESULTS ===")
    print("Accuracy:", accuracy_score(y_dev, y_pred_dev))
    print(classification_report(y_dev, y_pred_dev, zero_division=0))
    print("Macro F1 (DEV):", f1_score(y_dev, y_pred_dev, average="macro"))
    print("Micro F1 (DEV):", f1_score(y_dev, y_pred_dev, average="micro"))

    y_pred_test = model.predict(X_test)
    print("\n=== TEST RESULTS ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test, zero_division=0))
    print("Macro F1 (TEST):", f1_score(y_test, y_pred_test, average="macro"))
    print("Micro F1 (TEST):", f1_score(y_test, y_pred_test, average="micro"))

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to file")