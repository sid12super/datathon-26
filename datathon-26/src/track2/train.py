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


def tune_threshold_global(val_probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """
    Sweep a single threshold to maximize Micro-F1 on validation (active labels only).
    Returns (best_threshold, best_micro_f1).  Used as fallback for labels with no
    positive val examples.
    """
    best_thr, best_micro = 0.25, -1.0
    for thr in np.linspace(0.05, 0.90, 86):
        pred = (val_probs >= thr).astype(int)
        micro = f1_score(y_true, pred, average="micro", zero_division=0)
        if micro > best_micro:
            best_micro = micro
            best_thr = float(thr)
    return best_thr, best_micro


def tune_thresholds_per_label(
    val_probs: np.ndarray,
    y_true: np.ndarray,
    active_labels: list[str],
    global_fallback_thr: float,
) -> tuple[list[float], float]:
    """
    EDA showed 2768x class imbalance — a single global threshold is suboptimal.
    Tune one threshold per label by maximising binary F1 on val independently.
    Labels with zero positive val examples fall back to the global threshold.
    Returns (per_label_thresholds, micro_f1_with_per_label_thrs).
    """
    sweep = np.linspace(0.05, 0.90, 86)
    thresholds: list[float] = []

    for i in range(len(active_labels)):
        n_pos = int(y_true[:, i].sum())
        if n_pos == 0:
            # Can't tune — no signal in val for this label
            thresholds.append(global_fallback_thr)
            continue
        best_thr, best_f1 = global_fallback_thr, -1.0
        for thr in sweep:
            pred = (val_probs[:, i] >= thr).astype(int)
            f1 = f1_score(y_true[:, i], pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        thresholds.append(best_thr)

    # Evaluate overall Micro-F1 with the assembled per-label thresholds
    pred_matrix = np.column_stack([
        (val_probs[:, i] >= thresholds[i]).astype(int)
        for i in range(len(active_labels))
    ])
    micro = f1_score(y_true, pred_matrix, average="micro", zero_division=0)
    return thresholds, micro


def report_per_label_f1(
    val_probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: list[float],
    active_labels: list[str],
    n_show: int = 15,
) -> None:
    """Print per-label binary F1 on val — useful for spotting which labels are hard."""
    rows = []
    for i, label in enumerate(active_labels):
        pred = (val_probs[:, i] >= thresholds[i]).astype(int)
        n_pos = int(y_true[:, i].sum())
        f1 = f1_score(y_true[:, i], pred, average="binary", zero_division=0)
        rows.append((label, f1, n_pos, thresholds[i]))

    rows.sort(key=lambda x: x[1])
    print(f"\n{'─'*65}")
    print(f"  Per-label val binary F1  (bottom {n_show})")
    print(f"{'─'*65}")
    print(f"  {'Label':<22} {'F1':>6}  {'val+':>5}  {'thr':>5}")
    for label, f1, pos, thr in rows[:n_show]:
        print(f"  {label:<22} {f1:>6.3f}  {pos:>5}  {thr:>5.2f}")
    print(f"\n  Per-label val binary F1  (top {n_show})")
    print(f"{'─'*65}")
    print(f"  {'Label':<22} {'F1':>6}  {'val+':>5}  {'thr':>5}")
    for label, f1, pos, thr in rows[-n_show:]:
        print(f"  {label:<22} {f1:>6.3f}  {pos:>5}  {thr:>5.2f}")
    print(f"{'─'*65}")


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

    # ── Threshold tuning ────────────────────────────────────────────────────
    # Global threshold (micro-F1 optimal) — used as fallback for labels with
    # no positive val examples (EDA: ~20 labels appear only once in train).
    global_thr, global_micro = tune_threshold_global(val_probs, Y_val_active)
    print(f"Global threshold  → val Micro-F1: {global_micro:.4f} @ thr={global_thr:.2f}")

    # Per-label thresholds — EDA showed 2768x imbalance, so different labels
    # naturally sit at very different operating points.
    per_label_thrs, per_label_micro = tune_thresholds_per_label(
        val_probs, Y_val_active, active_labels.tolist(), global_thr
    )
    print(f"Per-label threshold → val Micro-F1: {per_label_micro:.4f}"
          f"  (Δ {per_label_micro - global_micro:+.4f} vs global)")

    report_per_label_f1(val_probs, Y_val_active, per_label_thrs, active_labels.tolist())

    # ── Save artifacts ───────────────────────────────────────────────────────
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "ovr_logreg.joblib"))

    meta = {
        # Per-label thresholds (primary — aligned to active_labels order)
        "thresholds": per_label_thrs,
        # Global threshold kept for reference / backward compat
        "threshold": global_thr,
        "labels": labels,                      # includes "none"
        "active_labels": active_labels.tolist(),  # excludes "none"
        "vectorizer_mode": VECTORIZER_MODE,
        "C": C_VALUE,
        "solver": SOLVER,
        "max_iter": MAX_ITER,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model artifacts to: {MODEL_DIR}")
    print(" - model/tfidf.joblib")
    print(" - model/ovr_logreg.joblib")
    print(" - model/meta.json")


if __name__ == "__main__":
    main()