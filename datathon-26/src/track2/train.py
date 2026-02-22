# train.py
# Category-2 compliant: TF-IDF (word bigrams + char trigrams) + OVR LR & LinearSVC ensemble
# Run from src/track2/  →  saves artifacts to model/

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR       = "model/"
TRAIN_PATH      = "train.csv"
VAL_PATH        = "val.csv"
LABEL_LIST_PATH = "label_list.txt"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
C_LR      = 4.0
C_SVC     = 0.5
LR_SOLVER = "liblinear"
LR_MAX_ITER = 3000
SVC_MAX_ITER = 3000


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_label_list(path: str) -> list[str]:
    """Load official labels (lowercase); ensure 'none' is present; de-dup."""
    labels = [l.strip().lower() for l in open(path, "r", encoding="utf-8") if l.strip()]
    if "none" not in labels:
        labels.append("none")
    seen: set[str] = set()
    return [x for x in labels if not (x in seen or seen.add(x))]


def parse_topics(x: str, known_labels: set | None = None) -> list[str]:
    """Parse pipe-separated labels; filter unknowns if known_labels provided."""
    x = (x or "none").strip().lower()
    parts = [t.strip().lower() for t in x.split("|") if t.strip()]
    if known_labels is not None:
        parts = [t for t in parts if t in known_labels]
    return parts if parts else ["none"]


def combine_text(df: pd.DataFrame) -> list[str]:
    """Concatenate title + body — avoids empty-body articles losing all signal."""
    title = df["title"].fillna("").astype(str)
    body  = df["text"].fillna("").astype(str)
    return (title + ". " + body).tolist()


def build_union_vectorizer() -> FeatureUnion:
    """
    FeatureUnion of word bigrams + char 3-5-grams.
    Word TF-IDF handles semantic content; char TF-IDF captures ticker symbols,
    morphology, and spelling variation. Combined, they complement each other.
    """
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=150_000,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        max_features=100_000,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def tune_threshold_global(
    val_probs: np.ndarray, y_true: np.ndarray
) -> tuple[float, float]:
    """Sweep a single global threshold; maximise Micro-F1 on validation."""
    best_thr, best_micro = 0.25, -1.0
    for thr in np.linspace(0.05, 0.90, 86):
        pred  = (val_probs >= thr).astype(int)
        micro = f1_score(y_true, pred, average="micro", zero_division=0)
        if micro > best_micro:
            best_micro = micro
            best_thr   = float(thr)
    return best_thr, best_micro


def tune_thresholds_per_label(
    val_probs: np.ndarray,
    y_true: np.ndarray,
    active_labels: list[str],
    global_fallback_thr: float,
) -> tuple[list[float], float]:
    """
    Per-label binary-F1 threshold sweep on val.
    Labels with zero positive val examples fall back to the global threshold.
    NOTE: historically this overfits val (many rare-label thresholds collapse
    to 0.05); the global threshold generalises better — use for reference only.
    """
    sweep = np.linspace(0.05, 0.90, 86)
    thresholds: list[float] = []
    for i in range(len(active_labels)):
        n_pos = int(y_true[:, i].sum())
        if n_pos == 0:
            thresholds.append(global_fallback_thr)
            continue
        best_thr, best_f1 = global_fallback_thr, -1.0
        for thr in sweep:
            pred = (val_probs[:, i] >= thr).astype(int)
            f1   = f1_score(y_true[:, i], pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_thr = float(thr)
        thresholds.append(best_thr)

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
    n_show: int = 10,
) -> None:
    rows = []
    for i, label in enumerate(active_labels):
        pred  = (val_probs[:, i] >= thresholds[i]).astype(int)
        n_pos = int(y_true[:, i].sum())
        f1    = f1_score(y_true[:, i], pred, average="binary", zero_division=0)
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_PATH, dtype=str).fillna("")
    val_df   = pd.read_csv(VAL_PATH,   dtype=str).fillna("")

    labels      = load_label_list(LABEL_LIST_PATH)
    known_set   = set(labels)

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Labels: {len(labels)}")
    empty_body = (train_df["text"].str.strip() == "").sum()
    print(f"Train articles with empty body: {empty_body:,} → using title+body combined text")

    # ── Labels ────────────────────────────────────────────────────────────────
    y_train_sets = train_df["topics"].apply(lambda x: parse_topics(x, known_set)).tolist()
    y_val_sets   = val_df["topics"].apply(lambda x: parse_topics(x, known_set)).tolist()

    mlb     = MultiLabelBinarizer(classes=labels)
    Y_train = mlb.fit_transform(y_train_sets)
    Y_val   = mlb.transform(y_val_sets)

    label_mask     = np.array([c != "none" for c in mlb.classes_])
    active_labels  = mlb.classes_[label_mask].tolist()
    Y_train_active = Y_train[:, label_mask]
    Y_val_active   = Y_val[:, label_mask]

    # ── Features ──────────────────────────────────────────────────────────────
    train_texts = combine_text(train_df)
    val_texts   = combine_text(val_df)

    print("\nBuilding FeatureUnion (word bigrams + char 3-5-grams)…")
    vectorizer = build_union_vectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)
    print(f"Feature matrix: {X_train.shape[0]:,} × {X_train.shape[1]:,}")

    # ── Model 1: Logistic Regression ──────────────────────────────────────────
    print("\n=== Training Logistic Regression (OVR, liblinear) ===")
    base_lr = LogisticRegression(
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        C=C_LR,
        class_weight="balanced",
    )
    lr_clf = OneVsRestClassifier(base_lr)
    lr_clf.fit(X_train, Y_train_active)
    lr_probs = lr_clf.predict_proba(X_val)

    lr_global_thr, lr_global_micro = tune_threshold_global(lr_probs, Y_val_active)
    print(f"LR  global threshold → val Micro-F1: {lr_global_micro:.4f} @ thr={lr_global_thr:.2f}")

    # ── Model 2: LinearSVC (sigmoid-calibrated) ───────────────────────────────
    print("\n=== Training LinearSVC (OVR, sigmoid calibration) ===")
    # dual=False is faster when n_features > n_samples (our case: 250k features, 15k samples)
    base_svc = LinearSVC(
        C=C_SVC,
        max_iter=SVC_MAX_ITER,
        class_weight="balanced",
        dual=False,
    )
    svc_clf = OneVsRestClassifier(base_svc)
    svc_clf.fit(X_train, Y_train_active)
    # sigmoid of decision_function scores gives calibrated pseudo-probs
    svc_scores = svc_clf.decision_function(X_val)
    svc_probs  = sigmoid(svc_scores)

    svc_global_thr, svc_global_micro = tune_threshold_global(svc_probs, Y_val_active)
    print(f"SVC global threshold → val Micro-F1: {svc_global_micro:.4f} @ thr={svc_global_thr:.2f}")

    # ── Ensemble (average probs) ───────────────────────────────────────────────
    print("\n=== Ensemble (0.5 × LR + 0.5 × SVC) ===")
    ens_probs = 0.5 * lr_probs + 0.5 * svc_probs
    ens_global_thr, ens_global_micro = tune_threshold_global(ens_probs, Y_val_active)
    print(f"ENS global threshold → val Micro-F1: {ens_global_micro:.4f} @ thr={ens_global_thr:.2f}")

    # Per-label on ensemble (reported for reference; usually overfits val)
    per_label_thrs, per_label_micro = tune_thresholds_per_label(
        ens_probs, Y_val_active, active_labels, ens_global_thr
    )
    print(f"ENS per-label thresholds → val Micro-F1: {per_label_micro:.4f}"
          f"  (Δ {per_label_micro - ens_global_micro:+.4f} vs global)")

    # Use global threshold for ensemble (per-label tends to overfit on val)
    report_per_label_f1(ens_probs, Y_val_active, [ens_global_thr] * len(active_labels), active_labels)

    print("\n" + "="*55)
    print(f"  LR  alone  → val Micro-F1: {lr_global_micro:.4f}")
    print(f"  SVC alone  → val Micro-F1: {svc_global_micro:.4f}")
    print(f"  Ensemble   → val Micro-F1: {ens_global_micro:.4f}  ← SELECTED")
    print("="*55)

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf.joblib"))
    joblib.dump(lr_clf,     os.path.join(MODEL_DIR, "ovr_logreg.joblib"))
    joblib.dump(svc_clf,    os.path.join(MODEL_DIR, "ovr_svc.joblib"))

    meta = {
        # Primary threshold (global, ensemble)
        "threshold":       ens_global_thr,
        # Per-label kept for reference — not used by default
        "thresholds":      per_label_thrs,
        # Label info
        "labels":          labels,
        "active_labels":   active_labels,
        # Ensemble config
        "use_ensemble":    True,
        "lr_weight":       0.5,
        "svc_weight":      0.5,
        # Val metrics
        "val_micro_lr":    round(lr_global_micro,  4),
        "val_micro_svc":   round(svc_global_micro, 4),
        "val_micro_ens":   round(ens_global_micro, 4),
        # Hyper-params
        "C_LR":            C_LR,
        "C_SVC":           C_SVC,
        "solver":          LR_SOLVER,
        "max_iter_lr":     LR_MAX_ITER,
        "max_iter_svc":    SVC_MAX_ITER,
        "vectorizer":      "FeatureUnion(word-1-2, char_wb-3-5)",
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved artifacts to {MODEL_DIR}")
    print("  tfidf.joblib      (FeatureUnion: word + char)")
    print("  ovr_logreg.joblib (LogisticRegression)")
    print("  ovr_svc.joblib    (LinearSVC)")
    print("  meta.json")


if __name__ == "__main__":
    main()
