"""
Track 2 – predict.py  (BiLSTM + TF-IDF ensemble, Category 1)
=============================================================
Self-evaluate on val set:
    INPUT_PATH  = "val.csv"
    LABELS_PATH = "val.csv"

Final submission (set before zipping):
    INPUT_PATH  = "test.csv"
    LABELS_PATH = None

Before running inference, ensure model/meta_ensemble.json exists:
    python tune_ensemble.py   ← run once after training
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

# ==============================================================================
# CHANGE THESE PATHS IF NEEDED
# ==============================================================================

INPUT_PATH  = "test.csv"
OUTPUT_PATH = "predictions.csv"
LABELS_PATH = None
MODEL_PATH  = "model/"

# ==============================================================================
# YOUR CODE — IMPLEMENT THESE TWO FUNCTIONS
# ==============================================================================

import os
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import joblib


# ── Sigmoid helper ─────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ── BiLSTM model definition (must match bilstm_train.py exactly) ───────────────

class BiLSTMClassifier(nn.Module):
    """2-layer BiLSTM with masked mean-pooling. Matches bilstm_train.py."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        hidden_dim: int,
        num_layers: int,
        num_labels: int,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers    = num_layers,
            bidirectional = True,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask    = (x != 0).float()
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        emb         = self.dropout(self.embedding(x))
        out, _      = self.lstm(emb)
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths
        pooled = self.dropout(pooled)
        return self.fc(pooled)


# ── Tokenisation (identical to bilstm_train.py) ────────────────────────────────

def _tokenize_batch(texts: list, vocab: dict, max_len: int) -> torch.Tensor:
    UNK = vocab.get("<UNK>", 1)
    seqs = []
    for t in texts:
        tokens = t.lower().split()[:max_len]
        ids    = [vocab.get(tok, UNK) for tok in tokens]
        ids   += [0] * (max_len - len(ids))
        seqs.append(ids)
    return torch.tensor(seqs, dtype=torch.long)


# ── load_model ─────────────────────────────────────────────────────────────────

def load_model() -> dict:
    # ── BiLSTM ────────────────────────────────────────────────────────────────
    with open(os.path.join(MODEL_PATH, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(MODEL_PATH, "meta_bilstm.json"), "r") as f:
        meta_b = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BiLSTMClassifier(
        vocab_size = meta_b["vocab_size"],
        embed_dim  = meta_b["embed_dim"],
        hidden_dim = meta_b["hidden_dim"],
        num_layers = meta_b["num_layers"],
        num_labels = meta_b["num_labels"],
        dropout    = meta_b.get("dropout", 0.3),
    ).to(device)
    state = torch.load(
        os.path.join(MODEL_PATH, "bilstm.pt"),
        map_location=device, weights_only=True,
    )
    net.load_state_dict(state)
    net.eval()

    # ── TF-IDF ensemble ───────────────────────────────────────────────────────
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidf.joblib"))
    lr_clf     = joblib.load(os.path.join(MODEL_PATH, "ovr_logreg.joblib"))
    svc_clf    = joblib.load(os.path.join(MODEL_PATH, "ovr_svc.joblib"))

    # ── Ensemble meta (threshold + alpha) ─────────────────────────────────────
    meta_ens_path = os.path.join(MODEL_PATH, "meta_ensemble.json")
    if not os.path.exists(meta_ens_path):
        raise FileNotFoundError(
            f"{meta_ens_path} not found. Run `python tune_ensemble.py` first."
        )
    with open(meta_ens_path) as f:
        meta_ens = json.load(f)

    active_labels  = meta_ens["active_labels"]    # 115 labels (no "none")
    bilstm_alpha   = float(meta_ens["bilstm_alpha"])
    threshold      = float(meta_ens["threshold"])
    bilstm_classes = meta_b["labels"]             # 116 labels (from meta_bilstm.json)
    b_idx = [bilstm_classes.index(l) for l in active_labels]  # alignment

    print(
        f"Ensemble loaded  →  bilstm_alpha={bilstm_alpha}  "
        f"thr={threshold:.2f}  val_f1={meta_ens['val_micro_f1']}  device={device}"
    )

    return {
        # BiLSTM
        "net":           net,
        "vocab":         vocab,
        "max_len":       int(meta_b["max_len"]),
        "device":        device,
        "b_idx":         b_idx,
        # TF-IDF
        "vectorizer":    vectorizer,
        "lr_clf":        lr_clf,
        "svc_clf":       svc_clf,
        # Ensemble
        "active_labels": active_labels,
        "bilstm_alpha":  bilstm_alpha,
        "threshold":     threshold,
    }


# ── predict ────────────────────────────────────────────────────────────────────

def predict(model: dict, texts: list) -> list:
    """
    texts  — list of body strings passed by the harness.
    We re-read INPUT_PATH to combine title+body (matching training setup).
    Falls back to body-only if reading fails.
    """
    # Combine title + body to reproduce the exact text seen during training
    combined = texts
    try:
        df_full = pd.read_csv(INPUT_PATH, dtype=str).fillna("")
        if "title" in df_full.columns and len(df_full) == len(texts):
            combined = (df_full["title"] + ". " + df_full["text"]).tolist()
    except Exception:
        pass  # silently fall back to body-only

    net          = model["net"]
    vocab        = model["vocab"]
    max_len      = model["max_len"]
    device       = model["device"]
    b_idx        = model["b_idx"]
    vectorizer   = model["vectorizer"]
    lr_clf       = model["lr_clf"]
    svc_clf      = model["svc_clf"]
    active_labels = model["active_labels"]
    bilstm_alpha  = model["bilstm_alpha"]
    threshold     = model["threshold"]

    # ── BiLSTM probs (N, 116) → slice to 115 active labels ───────────────────
    INFER_BATCH = 256
    all_probs: list = []
    with torch.no_grad():
        for i in range(0, len(combined), INFER_BATCH):
            batch  = combined[i : i + INFER_BATCH]
            X      = _tokenize_batch(batch, vocab, max_len).to(device)
            logits = net(X)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    B_full = np.vstack(all_probs)          # (N, 116)
    B      = B_full[:, b_idx]              # (N, 115)

    # ── TF-IDF probs (N, 115) ─────────────────────────────────────────────────
    X_tfidf   = vectorizer.transform(combined)
    lr_probs  = lr_clf.predict_proba(X_tfidf)              # (N, 115)
    svc_probs = _sigmoid(svc_clf.decision_function(X_tfidf))  # (N, 115)
    T         = 0.5 * lr_probs + 0.5 * svc_probs           # (N, 115)

    # ── Ensemble + threshold ──────────────────────────────────────────────────
    ens_probs = bilstm_alpha * B + (1.0 - bilstm_alpha) * T
    preds_bin = (ens_probs >= threshold).astype(int)

    out: list = []
    for row in preds_bin:
        label_list = [active_labels[j] for j, v in enumerate(row) if v == 1]
        out.append("|".join(label_list) if label_list else "none")

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
