import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

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
        ("vectorizer", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ("classifier", LogisticRegression(solver="liblinear", class_weight="balanced"))
    ])
    model.fit(X_train, y_train)

    y_pred_dev = model.predict(X_dev)
    print("\n=== DEV RESULTS ===")
    print("Accuracy:", accuracy_score(y_dev, y_pred_dev))
    print(classification_report(y_dev, y_pred_dev, zero_division=0))

    y_pred_test = model.predict(X_test)
    print("\n=== TEST RESULTS ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test, zero_division=0))