import os
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

def load_data(file):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    df["text"] = df["text"].fillna("")
    return df["text"], df["label"]

def evaluate_model(name, model, X_train, y_train, X_dev, y_dev, X_test, y_test):
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)

    y_dev_pred = model.predict(X_dev)
    print("\nDEV RESULTS:")
    print("Accuracy:", accuracy_score(y_dev, y_dev_pred))
    print("Macro F1:", f1_score(y_dev, y_dev_pred, average="macro"))
    print(classification_report(y_dev, y_dev_pred, zero_division=0))

    y_test_pred = model.predict(X_test)
    print("\nTEST RESULTS:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Macro F1:", f1_score(y_test, y_test_pred, average="macro"))
    print(classification_report(y_test, y_test_pred, zero_division=0))

    return y_test_pred


if __name__ == "__main__":
    X_train, y_train = load_data("train_tokenized.tsv")
    X_dev, y_dev = load_data("dev_tokenized.tsv")
    X_test, y_test = load_data("test_tokenized.tsv")
    nb_model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])

    y_pred = evaluate_model("Naive Bayes", nb_model, X_train, y_train, X_dev, y_dev, X_test, y_test)
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Naive Bayes)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_nb.png")
    print("\nConfusion matrix saved to file")