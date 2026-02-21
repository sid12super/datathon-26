# train.py
# Baseline (Category-2 compliant): TF-IDF + One-vs-Rest Logistic Regression
# - Trains in src/track2/ (recommended)
# - Saves artifacts to model/ for predict.py to load

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


# -----------------------------
# Paths (keep these if you run from src/track2)
# -----------------------------
MODEL_DIR = "model/"
TRAIN_PATH = "train.csv"
VAL_PATH = "val.csv"
LABEL_LIST_PATH = "label_list.txt"


# -----------------------------
# Config (tweak if needed)
# -----------------------------
# Choose ONE vectorizer mode
VECTORIZER_MODE = "word"   # "word" or "char"

# Logistic Regression hyperparams
C_VALUE = 4.0
MAX_ITER = 3000
SOLVER = "liblinear"       # "liblinear" is stable; "saga" can be faster for big data


def parse_topics(x: str) -> list[str]:
    """Parse pipe-separated labels; normalize to lowercase; fallback to ['none']."""
    x = (x or "none").strip().lower()
    parts = [t.strip().lower() for t in x.split("|") if t.strip()]
    return parts if parts else ["none"]


def load_label_list(path: str) -> list[str]:
    """Load official labels and normalize to lowercase; ensure 'none' exists."""
    labels = [l.strip().lower() for l in open(path, "r", encoding="utf-8") if l.strip()]
    if "none" not in labels:
        labels.append("none")
    # de-dup while preserving order
    seen = set()
    labels = [x for x in labels if not (x in seen or seen.add(x))]
    return labels


def build_vectorizer(mode: str) -> TfidfVectorizer:
    if mode == "char":
        # Often strong for noisy text / misspellings
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            max_features=300_000,
            sublinear_tf=True
        )
    # default word mode
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=200_000,
        sublinear_tf=True
    )


def tune_threshold(val_probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """
    Sweep thresholds to maximize Micro-F1 on validation (active labels only).
    Returns (best_threshold, best_micro_f1).
    """
    best_thr, best_micro = 0.25, -1.0
    for thr in np.linspace(0.05, 0.60, 56):  # 0.05..0.60 step ~0.01
        pred = (val_probs >= thr).astype(int)
        micro = f1_score(y_true, pred, average="micro", zero_division=0)
        if micro > best_micro:
            best_micro = micro
            best_thr = float(thr)
    return best_thr, best_micro


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    train = pd.read_csv(TRAIN_PATH, dtype=str).fillna("")
    val = pd.read_csv(VAL_PATH, dtype=str).fillna("")

    # Labels
    labels = load_label_list(LABEL_LIST_PATH)

    # Parse topics
    y_train_sets = train["topics"].apply(parse_topics).tolist()
    y_val_sets = val["topics"].apply(parse_topics).tolist()

    # MultiLabelBinarizer with fixed classes ensures consistent ordering at inference
    mlb = MultiLabelBinarizer(classes=labels)
    Y_train = mlb.fit_transform(y_train_sets)
    Y_val = mlb.transform(y_val_sets)

    # Treat "none" as a fallback label (not actively predicted)
    label_mask = np.array([c != "none" for c in mlb.classes_])
    active_labels = mlb.classes_[label_mask]
    Y_train_active = Y_train[:, label_mask]
    Y_val_active = Y_val[:, label_mask]

    # Vectorize text
    vectorizer = build_vectorizer(VECTORIZER_MODE)
    X_train = vectorizer.fit_transform(train["text"].astype(str).tolist())
    X_val = vectorizer.transform(val["text"].astype(str).tolist())

    # Model
    base_lr = LogisticRegression(
        solver=SOLVER,
        max_iter=MAX_ITER,
        C=C_VALUE,
        class_weight="balanced"
    )
    if SOLVER == "saga":
        # saga supports n_jobs in recent sklearn
        base_lr.set_params(n_jobs=-1)

    clf = OneVsRestClassifier(base_lr)

    clf.fit(X_train, Y_train_active)

    # Validation probabilities (for threshold tuning)
    if hasattr(clf, "predict_proba"):
        val_probs = clf.predict_proba(X_val)
    else:
        # Fallback (rare): convert decision_function to pseudo-prob via sigmoid
        scores = clf.decision_function(X_val)
        val_probs = 1.0 / (1.0 + np.exp(-scores))

    # Tune threshold
    best_thr, best_micro = tune_threshold(val_probs, Y_val_active)
    print(f"Best val Micro-F1 (active labels): {best_micro:.4f} @ thr={best_thr:.2f}")

    # Save artifacts
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "ovr_logreg.joblib"))

    meta = {
        "threshold": best_thr,
        "labels": labels,  # includes "none"
        "active_labels": active_labels.tolist(),  # excludes "none"
        "vectorizer_mode": VECTORIZER_MODE,
        "C": C_VALUE,
        "solver": SOLVER,
        "max_iter": MAX_ITER,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model artifacts to: {MODEL_DIR}")
    print("Expected files:")
    print(" - model/tfidf.joblib")
    print(" - model/ovr_logreg.joblib")
    print(" - model/meta.json")


if __name__ == "__main__":
    main()