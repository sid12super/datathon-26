"""
tune_ensemble.py  — run from src/track2/
Sweeps (bilstm_alpha, threshold) on val.csv to find the best ensemble of
BiLSTM + TF-IDF (LR + SVC). Saves result to model/meta_ensemble.json.

Usage:
    python tune_ensemble.py
"""
import os, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

MODEL_PATH = "model/"
VAL_PATH   = "val.csv"

# ── helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_labels, dropout=0.0):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                  bidirectional=True, batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask    = (x != 0).float()
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        emb     = self.dropout(self.embedding(x))
        out, _  = self.lstm(emb)
        pooled  = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths
        pooled  = self.dropout(pooled)
        return self.fc(pooled)


def _tokenize(texts: list, vocab: dict, max_len: int) -> torch.Tensor:
    UNK = vocab.get("<UNK>", 1)
    seqs = []
    for t in texts:
        ids  = [vocab.get(tok, UNK) for tok in t.lower().split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        seqs.append(ids)
    return torch.tensor(seqs, dtype=torch.long)


def _bilstm_probs(texts, vocab, net, max_len, device, bs=256) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            X = _tokenize(texts[i:i+bs], vocab, max_len).to(device)
            parts.append(torch.sigmoid(net(X)).cpu().numpy())
    return np.vstack(parts)  # (N, num_labels)


def _tfidf_probs(texts, vec, lr, svc) -> np.ndarray:
    X         = vec.transform(texts)
    lr_probs  = lr.predict_proba(X)                   # (N, 115)
    svc_probs = _sigmoid(svc.decision_function(X))    # (N, 115)
    return 0.5 * lr_probs + 0.5 * svc_probs           # (N, 115)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Load BiLSTM ──────────────────────────────────────────────────────────
    print("Loading BiLSTM…")
    with open(os.path.join(MODEL_PATH, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(MODEL_PATH, "meta_bilstm.json")) as f:
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
    net.load_state_dict(torch.load(
        os.path.join(MODEL_PATH, "bilstm.pt"),
        map_location=device, weights_only=True,
    ))
    net.eval()
    print(f"  hidden={meta_b['hidden_dim']}  val_f1={meta_b['val_micro_f1']}  "
          f"thr={meta_b['best_threshold']:.2f}  device={device}")

    # ── Load TF-IDF ensemble ─────────────────────────────────────────────────
    print("Loading TF-IDF models…")
    vec = joblib.load(os.path.join(MODEL_PATH, "tfidf.joblib"))
    lr  = joblib.load(os.path.join(MODEL_PATH, "ovr_logreg.joblib"))
    svc = joblib.load(os.path.join(MODEL_PATH, "ovr_svc.joblib"))
    with open(os.path.join(MODEL_PATH, "meta.json")) as f:
        meta_t = json.load(f)
    active_labels = meta_t["active_labels"]   # 115 labels (no "none")
    print(f"  TF-IDF alone → val Micro F1: {meta_t['val_micro_ens']:.4f}")

    # ── Label alignment ──────────────────────────────────────────────────────
    bilstm_classes = meta_b["labels"]                          # 116 labels (from meta_bilstm.json)
    b_idx = [bilstm_classes.index(l) for l in active_labels]  # 115 positions

    # ── Val data ─────────────────────────────────────────────────────────────
    print("Preparing val data…")
    val_df   = pd.read_csv(VAL_PATH, dtype=str).fillna("")
    combined = (val_df["title"] + ". " + val_df["text"]).tolist()

    def parse(x):
        parts = [t.strip().lower() for t in (x or "none").split("|") if t.strip()]
        return parts if parts else ["none"]

    y_sets   = val_df["topics"].apply(parse).tolist()
    y_active = [[l for l in s if l in set(active_labels)] for s in y_sets]
    tmp_mlb  = MultiLabelBinarizer(classes=active_labels)
    Y_active = tmp_mlb.fit_transform(y_active)            # (N, 115)
    none_true = (Y_active.sum(axis=1) == 0).astype(int)  # (N,) — 1 for "none" articles

    # ── Inference ────────────────────────────────────────────────────────────
    print("BiLSTM inference…")
    B_full = _bilstm_probs(combined, vocab, net, meta_b["max_len"], device)  # (N, 116)
    B      = B_full[:, b_idx]                                                 # (N, 115)

    print("TF-IDF inference…")
    T = _tfidf_probs(combined, vec, lr, svc)                                  # (N, 115)

    # ── Sweep (alpha, threshold) ─────────────────────────────────────────────
    print("\nSweeping bilstm_alpha ∈ {0.2, 0.3, 0.4, 0.5, 0.6}  ×  thr ∈ [0.05, 0.90]…")
    best_micro, best_alpha, best_thr = 0.0, 0.5, 0.25
    thresholds = np.linspace(0.05, 0.90, 86)

    for alpha in [0.2, 0.3, 0.4, 0.5, 0.6]:
        ens = alpha * B + (1.0 - alpha) * T
        for thr in thresholds:
            pred115   = (ens >= thr).astype(int)
            none_pred = (pred115.sum(axis=1) == 0).astype(int)
            # 116-label Micro F1 to match predict.py harness scoring
            Y116  = np.column_stack([Y_active, none_true])
            P116  = np.column_stack([pred115,  none_pred])
            micro = f1_score(Y116, P116, average="micro", zero_division=0)
            if micro > best_micro:
                best_micro = micro
                best_alpha = alpha
                best_thr   = float(thr)

    delta = best_micro - meta_t["val_micro_ens"]
    print(f"\n  Best ensemble → bilstm_alpha={best_alpha}  thr={best_thr:.2f}  "
          f"Micro F1={best_micro:.4f}  (Δ {delta:+.4f} vs TF-IDF)")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {
        "bilstm_alpha":  best_alpha,
        "tfidf_alpha":   round(1.0 - best_alpha, 1),
        "threshold":     round(best_thr, 4),
        "val_micro_f1":  round(best_micro, 4),
        "active_labels": active_labels,
    }
    out_path = os.path.join(MODEL_PATH, "meta_ensemble.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {out_path}")
    print("\nNext step: python predict.py")


if __name__ == "__main__":
    main()
