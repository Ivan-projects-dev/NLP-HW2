import os
import pandas as pd
from collections import Counter

def load_df(file):
    df = pd.read_csv(file, sep="\t", header=None, names=["text", "label"])
    df["text"] = df["text"].fillna("")
    return df

def class_distribution(file):
    df = load_df(file)
    print("\nClass distribution:")
    print(df["label"].value_counts()) # num of samples per class

def top_unigrams(file, top_n=10):
    df = load_df(file)
    print("\nTop words per class:\n")
  
    for label in df["label"].unique():
        words = []
        texts = df[df["label"] == label]["text"]
        
        for t in texts:
            words.extend(str(t).split()) # split text into words and add to list
        
        counter = Counter(words)
        print(f"{label}: {counter.most_common(top_n)}\n")

def top_bigrams(file, top_n=10):
    df = load_df(file)
    print("\nTop bigrams per class:\n")
    
    for label in df["label"].unique():
        bigrams = []
        texts = df[df["label"] == label]["text"]
        
        for t in texts:
            words = str(t).split()
            bigrams.extend(zip(words, words[1:]))
        
        counter = Counter(bigrams)
        print(f"{label}: {counter.most_common(top_n)}\n") 

def vocabulary_size(file):
    df = load_df(file)
    vocab = set()
    
    for t in df["text"]:
        vocab.update(t.split())
    
    print(f"\nVocabulary size: {len(vocab)}") # total num of unique tokens

if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__))) # ensures script works regardless of directory
    class_distribution("train_tokenized.tsv")
    
    top_unigrams("train_tokenized.tsv")
    top_bigrams("train_tokenized.tsv")
    vocabulary_size("train_tokenized.tsv")