"""
Track 2 – predict.py
=====================
Implement load_model() and predict() only.
DO NOT modify anything below the marked line.

Self-evaluate on val set:
    INPUT_PATH  = "val.csv"
    LABELS_PATH = "val.csv"

Final submission (paths must be set to test before submitting):
    INPUT_PATH  = "test.csv"
    LABELS_PATH = None
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

# ==============================================================================
# CHANGE THESE PATHS IF NEEDED
# ==============================================================================

INPUT_PATH  = "val.csv"
OUTPUT_PATH = "predictions.csv"
LABELS_PATH = "val.csv"            # set to "val.csv" for self-evaluation only
MODEL_PATH  = "model/"

# ==============================================================================
# YOUR CODE — IMPLEMENT THESE TWO FUNCTIONS
# ==============================================================================

def load_model():
    import os
    import json
    import joblib
    import numpy as np

    vec_path = os.path.join(MODEL_PATH, "tfidf.joblib")
    clf_path = os.path.join(MODEL_PATH, "ovr_logreg.joblib")
    meta_path = os.path.join(MODEL_PATH, "meta.json")

    vectorizer = joblib.load(vec_path)
    clf = joblib.load(clf_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    active_labels = meta["active_labels"]

    # Prefer per-label thresholds (written by updated train.py);
    # fall back to the single global threshold for older model artifacts.
    if "thresholds" in meta:
        thresholds = np.array(meta["thresholds"], dtype=float)
    else:
        thresholds = np.full(len(active_labels), float(meta["threshold"]))

    return {
        "vectorizer": vectorizer,
        "clf": clf,
        "thresholds": thresholds,          # shape: (n_active_labels,)
        "active_labels": active_labels,    # excludes "none"
    }


def predict(model, texts: list[str]) -> list[str]:
    import numpy as np

    vectorizer  = model["vectorizer"]
    clf         = model["clf"]
    thresholds  = model["thresholds"]   # per-label array
    active_labels = model["active_labels"]

    X = vectorizer.transform([t if isinstance(t, str) else "" for t in texts])

    probs = clf.predict_proba(X)           # (n_samples, n_active_labels)
    preds = (probs >= thresholds).astype(int)   # broadcast row-wise

    out = []
    for row in preds:
        labels = [active_labels[j] for j, v in enumerate(row) if v == 1]
        out.append("|".join(labels) if labels else "none")
    return out

# ==============================================================================
# DO NOT MODIFY ANYTHING BELOW THIS LINE
# ==============================================================================

def _parse_topics(series: pd.Series):
    return series.fillna("none").apply(
        lambda x: frozenset(t.strip() for t in x.split("|") if t.strip())
                  or frozenset(["none"])
    )

def _score(y_true_sets, y_pred_sets):
    mlb = MultiLabelBinarizer()
    mlb.fit(list(y_true_sets) + list(y_pred_sets))
    Y_true = mlb.transform(y_true_sets)
    Y_pred = mlb.transform(y_pred_sets)
    micro   = f1_score(Y_true, Y_pred, average="micro",   zero_division=0)
    macro   = f1_score(Y_true, Y_pred, average="macro",   zero_division=0)
    exact   = accuracy_score(Y_true, Y_pred)
    hamming = 1 - hamming_loss(Y_true, Y_pred)
    print("\n" + "="*45)
    print(f"  Micro F1 (PRIMARY) : {micro:.4f}")
    print(f"  Macro F1           : {macro:.4f}")
    print(f"  Exact-match acc    : {exact:.4f}")
    print(f"  Hamming accuracy   : {hamming:.4f}")
    print("="*45 + "\n")

def main():
    df = pd.read_csv(INPUT_PATH, dtype=str)
    missing = {"article_id", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    print(f"Loaded {len(df):,} articles from {INPUT_PATH}")

    model = load_model()
    preds = predict(model, df["text"].fillna("").tolist())

    if len(preds) != len(df):
        raise ValueError(f"predict() returned {len(preds)} predictions for {len(df)} articles.")

    out = df[["article_id"]].copy()
    out["topics"] = preds
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")

    if LABELS_PATH:
        gt     = pd.read_csv(LABELS_PATH, dtype=str)
        merged = gt.merge(out, on="article_id", suffixes=("_true", "_pred"))
        n_miss = len(gt) - len(merged)
        if n_miss > 0:
            print(f"WARNING: {n_miss} article_id(s) missing from predictions.")
        _score(
            _parse_topics(merged["topics_true"]),
            _parse_topics(merged["topics_pred"])
        )

if __name__ == "__main__":
    main()