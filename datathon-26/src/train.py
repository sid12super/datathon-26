import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

MODEL_DIR = "model/"
TRAIN_PATH = "train.csv"
VAL_PATH = "val.csv"
LABEL_LIST_PATH = "label_list.txt"

def parse_topics(x: str):
    x = (x or "none").strip()
    parts = [t.strip() for t in x.split("|") if t.strip()]
    return parts if parts else ["none"]

def load_label_list(path: str):
    labels = [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]
    # ensure "none" exists (it does in your file) :contentReference[oaicite:4]{index=4}
    if "none" not in labels:
        labels.append("none")
    return labels

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH, dtype=str).fillna("")
    val = pd.read_csv(VAL_PATH, dtype=str).fillna("")

    labels = load_label_list(LABEL_LIST_PATH)

    y_train_sets = train["topics"].apply(parse_topics).tolist()
    y_val_sets = val["topics"].apply(parse_topics).tolist()

    # MultiLabelBinarizer with fixed classes for consistent columns at inference
    mlb = MultiLabelBinarizer(classes=labels)
    Y_train = mlb.fit_transform(y_train_sets)
    Y_val = mlb.transform(y_val_sets)

    # IMPORTANT: treat "none" as a fallback label (not something we actively predict)
    # We'll train classifiers for all labels EXCEPT "none"
    label_mask = np.array([c != "none" for c in mlb.classes_])
    active_labels = mlb.classes_[label_mask]
    Y_train_active = Y_train[:, label_mask]
    Y_val_active = Y_val[:, label_mask]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=200_000,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(train["text"].tolist())
    X_val = vectorizer.transform(val["text"].tolist())

    clf = OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            C=4.0,
            class_weight="balanced"
        )
    )

    clf.fit(X_train, Y_train_active)

    # Choose a decent default threshold and report val micro-f1 (active labels only)
    val_probs = clf.predict_proba(X_val)
    thr = 0.25
    val_pred = (val_probs >= thr).astype(int)
    micro = f1_score(Y_val_active, val_pred, average="micro", zero_division=0)
    print(f"Validation Micro-F1 (active labels, thr={thr}): {micro:.4f}")

    # Save artifacts
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "ovr_logreg.joblib"))

    meta = {
        "threshold": thr,
        "labels": labels,
        "active_labels": active_labels.tolist()
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print("Saved model to model/")

if __name__ == "__main__":
    main()