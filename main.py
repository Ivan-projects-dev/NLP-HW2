import pandas as pd
import os
import re

def clean_all_files(file_names=("train.tsv", "dev.tsv", "test.tsv")):
    for file in file_names:
        df = pd.read_csv(file, sep="\t", header=None, names=["text", "label", "id"])
        df = df[~df["label"].astype(str).str.contains(",")]
        df = df[["text", "label"]]
        output_file = file.replace(".tsv", "_clean.tsv")
        df.to_csv(output_file, sep="\t", index=False, header=False)
        print(f"{file} -> {output_file} | {len(df)} rows")

def map_to_ekman(input_file, output_file):
    df = pd.read_csv(input_file, sep="\t", header=None, names=["text", "label"])

    id_to_emotion = {
        0: "admiration",
        1: "amusement", 
        2: "anger", 
        3: "annoyance",
        4: "approval", 
        5: "caring", 
        6: "confusion", 
        7: "curiosity",
        8: "desire", 
        9: "disappointment", 
        10: "disapproval", 
        11: "disgust",
        12: "embarrassment", 
        13: "excitement", 
        14: "fear", 
        15: "gratitude",
        16: "grief", 
        17: "joy", 
        18: "love", 
        19: "nervousness",
        20: "optimism", 
        21: "pride", 
        22: "realization", 
        23: "relief",
        24: "remorse", 
        25: "sadness", 
        26: "surprise", 
        27: "neutral"
    }

    ekman = {
        "anger": ["anger", "annoyance", "disapproval"],
        "disgust": ["disgust"],
        "fear": ["fear", "nervousness"],
        "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
        "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
        "surprise": ["surprise", "realization", "confusion", "curiosity"]
    }

    def map_label(label_id):
        emotion = id_to_emotion[int(label_id)]
        if emotion == "neutral":
            return "neutral"
        for k, v in ekman.items():
            if emotion in v:
                return k
        return None

    df["label"] = df["label"].apply(map_label)
    df = df[df["label"].notnull()]
    df.to_csv(output_file, sep="\t", index=False, header=False)
    print(f"{input_file} -> {output_file} | {len(df)} rows")

def tokenize(text):
    text = text.lower()
    text = re.sub(r"\d+", "NUM", text) # numbers are not needed so replace with token
    tokens = re.findall(r"\b\w+\b", text) # tokenization
    tokens = [t for t in tokens if len(t) > 1 or t == "I"] # remove all short words aside from I
    return tokens

def tokenize_file(input_file, output_file):
    df = pd.read_csv(input_file, sep="\t", header=None, names=["text", "label"])
    df["text"] = df["text"].apply(lambda x: " ".join(tokenize(x)))
    df.to_csv(output_file, sep="\t", index=False, header=False)
    print(f"{input_file} -> {output_file}")

if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    clean_all_files()

    map_to_ekman("train_clean.tsv", "train_ekman.tsv")
    map_to_ekman("dev_clean.tsv", "dev_ekman.tsv")
    map_to_ekman("test_clean.tsv", "test_ekman.tsv")

    tokenize_file("train_ekman.tsv", "train_tokenized.tsv")
    tokenize_file("dev_ekman.tsv", "dev_tokenized.tsv")
    tokenize_file("test_ekman.tsv", "test_tokenized.tsv")