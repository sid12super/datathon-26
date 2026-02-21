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

INPUT_PATH  = "test.csv"
OUTPUT_PATH = "predictions.csv"
LABELS_PATH = None             # set to "val.csv" for self-evaluation only
MODEL_PATH  = "model/"

# ==============================================================================
# YOUR CODE — IMPLEMENT THESE TWO FUNCTIONS
# ==============================================================================

def load_model():
    """
    Load and return your trained model from MODEL_PATH.

    Example (PyTorch / embedding model):
        import torch
        model = MyModel()
        model.load_state_dict(torch.load(MODEL_PATH + "weights.pt", map_location="cpu"))
        model.eval()
        return model

    Example (scikit-learn):
        import joblib
        return joblib.load(MODEL_PATH + "classifier.pkl")
    """
    raise NotImplementedError("Implement load_model()")


def predict(model, texts: list[str]) -> list[str]:
    """
    Run inference on a list of article texts.

    Args:
        model  : whatever load_model() returns
        texts  : list of raw article text strings

    Returns:
        A list of pipe-separated topic strings, one per article.
        e.g. ["earn", "money-fx|trade", "none", "earn|trade|grain"]

    Rules:
        - Return exactly len(texts) predictions
        - Multiple labels separated by pipe: "earn|trade"
        - Articles with no predicted label must return "none"
        - Use only labels from label_list.txt
        - Label order does not matter
    """
    raise NotImplementedError("Implement predict()")

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