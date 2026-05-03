import os
import pandas as pd
from collections import Counter

def class_distribution(file):
    df = pd.read_csv(file, sep="\t", header=None, names=["text", "label"])
    df["text"] = df["text"].fillna("")
    print("\nClass distribution:")
    print(df["label"].value_counts())

def top_unigrams(file, top_n=10):
    df = pd.read_csv(file, sep="\t", header=None, names=["text", "label"])
    df["text"] = df["text"].fillna("")
    print("\nTop words per class:\n")
  
    for label in df["label"].unique():
        words = []
        texts = df[df["label"] == label]["text"]
        
        for t in texts:
            words.extend(str(t).split())
        
        counter = Counter(words)
        print(f"{label}: {counter.most_common(top_n)}\n")

def top_bigrams(file, top_n=10):
    df = pd.read_csv(file, sep="\t", header=None, names=["text", "label"])
    df["text"] = df["text"].fillna("")
    print("\nTop bigrams per class:\n")
    
    for label in df["label"].unique():
        bigrams = []
        texts = df[df["label"] == label]["text"]
        
        for t in texts:
            words = t.split()
            bigrams.extend(zip(words, words[1:]))
        
        counter = Counter(bigrams)
        print(f"{label}: {counter.most_common(top_n)}\n")

def vocabulary_size(file):
    df = pd.read_csv(file, sep="\t", header=None, names=["text", "label"])
    df["text"] = df["text"].fillna("")
    vocab = set()
    
    for t in df["text"]:
        vocab.update(t.split())
    
    print(f"\nVocabulary size: {len(vocab)}")

if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    class_distribution("train_tokenized.tsv")
    
    top_unigrams("train_tokenized.tsv")
    top_bigrams("train_tokenized.tsv")
    vocabulary_size("train_tokenized.tsv")