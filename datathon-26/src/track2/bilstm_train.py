# bilstm_train.py
# Category-1: BiLSTM + GloVe (300d) multi-label text classifier
# Run from  src/track2/  →  python bilstm_train.py
# Saves artifacts to model/
#
# If val Micro F1 stays below 0.82 after 5 epochs, bump HIDDEN_DIM to 512.

import os
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

# ── Config ─────────────────────────────────────────────────────────────────────

TRAIN_PATH      = "train.csv"
VAL_PATH        = "val.csv"
LABEL_LIST_PATH = "label_list.txt"
GLOVE_PATH      = "data/glove/glove.6B.300d.txt"
MODEL_DIR       = "model/"

VOCAB_SIZE      = 50_000
MAX_LEN         = 300
EMBED_DIM       = 300

HIDDEN_DIM      = 256       # set to 512 if val Micro F1 < 0.82 after 5 epochs
NUM_LAYERS      = 2
DROPOUT         = 0.3

BATCH_SIZE      = 64
EPOCHS          = 10
LR              = 1e-3
PATIENCE        = 3
POS_WEIGHT_CAP  = 50.0      # cap pos_weight for very rare labels

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_label_list(path: str) -> list[str]:
    labels = [ln.strip().lower() for ln in open(path, encoding="utf-8") if ln.strip()]
    if "none" not in labels:
        labels.append("none")
    seen: set = set()
    return [x for x in labels if not (x in seen or seen.add(x))]


def parse_topics(x: str, known: set | None = None) -> list[str]:
    x = (x or "none").strip().lower()
    parts = [t.strip() for t in x.split("|") if t.strip()]
    if known is not None:
        parts = [t for t in parts if t in known]
    return parts if parts else ["none"]


def combine_text(df: pd.DataFrame) -> list[str]:
    title = df["title"].fillna("").astype(str)
    body  = df["text"].fillna("").astype(str)
    return (title + ". " + body).tolist()


def build_vocab(texts: list[str], max_vocab: int) -> dict[str, int]:
    from collections import Counter
    counter: Counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def texts_to_sequences(texts: list[str], vocab: dict, max_len: int) -> np.ndarray:
    UNK = vocab.get("<UNK>", 1)
    seqs = []
    for t in texts:
        tokens = t.lower().split()[:max_len]
        ids    = [vocab.get(tok, UNK) for tok in tokens]
        ids   += [0] * (max_len - len(ids))
        seqs.append(ids)
    return np.array(seqs, dtype=np.int64)


def load_glove(path: str, vocab: dict, embed_dim: int) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"GloVe file not found: {path}\n"
            f"Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/ "
            f"and unzip glove.6B.300d.txt to data/glove/"
        )
    print(f"Loading GloVe from {path} …")
    rng          = np.random.default_rng(42)
    embed_matrix = rng.uniform(-0.05, 0.05, (len(vocab), embed_dim)).astype(np.float32)
    embed_matrix[0] = 0.0   # <PAD> → zero vector
    found = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            if word in vocab:
                embed_matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f"  {found:,}/{len(vocab):,} vocab words matched in GloVe")
    return embed_matrix


# ── Dataset ────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(sequences)   # (N, max_len)  int64
        self.Y = torch.from_numpy(labels)      # (N, num_labels) float32

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# ── Model ──────────────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    2-layer BiLSTM with masked mean-pooling over all output timesteps.
    Mean-pooling is more robust than last-hidden-state for long sequences
    and captures signal from the full article rather than just the end.
    """

    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int,
        embed_matrix: np.ndarray,
        hidden_dim:   int,
        num_layers:   int,
        num_labels:   int,
        dropout:      float,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embed_matrix, dtype=torch.float32)
        )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers    = num_layers,
            bidirectional = True,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)  — 0 is the PAD token index
        mask    = (x != 0).float()                              # (B, L)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)   # (B, 1)

        emb         = self.dropout(self.embedding(x))           # (B, L, E)
        out, _      = self.lstm(emb)                            # (B, L, H*2)

        # Masked mean-pool: sum real-token hidden states / real-token count
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths  # (B, H*2)
        pooled = self.dropout(pooled)

        return self.classifier(pooled)                          # (B, num_labels)


# ── Evaluation & threshold sweep ───────────────────────────────────────────────

@torch.no_grad()
def get_val_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for X, Y in loader:
        logits = model(X.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(Y.numpy())
    return np.vstack(all_probs), np.vstack(all_labels)


def sweep_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    best_thr, best_micro = 0.5, -1.0
    for thr in np.arange(0.30, 0.95, 0.01):
        preds = (probs >= thr).astype(int)
        micro = f1_score(labels, preds, average="micro", zero_division=0)
        if micro > best_micro:
            best_micro = micro
            best_thr   = float(thr)
    return best_thr, best_micro


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device: {DEVICE}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_PATH, dtype=str).fillna("")
    val_df   = pd.read_csv(VAL_PATH,   dtype=str).fillna("")
    labels   = load_label_list(LABEL_LIST_PATH)
    known    = set(labels)
    num_labels = len(labels)
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Labels: {num_labels}")
    empty_body = (train_df["text"].str.strip() == "").sum()
    print(f"Train articles with empty body: {empty_body:,} (title used as signal)")

    # ── Labels ────────────────────────────────────────────────────────────────
    y_train_sets = train_df["topics"].apply(lambda x: parse_topics(x, known)).tolist()
    y_val_sets   = val_df["topics"].apply(lambda x: parse_topics(x, known)).tolist()

    mlb     = MultiLabelBinarizer(classes=labels)
    Y_train = mlb.fit_transform(y_train_sets).astype(np.float32)
    Y_val   = mlb.transform(y_val_sets).astype(np.float32)

    # ── Text ──────────────────────────────────────────────────────────────────
    train_texts = combine_text(train_df)
    val_texts   = combine_text(val_df)

    # ── Vocabulary ────────────────────────────────────────────────────────────
    print("Building vocabulary …")
    vocab = build_vocab(train_texts, VOCAB_SIZE)
    print(f"  Vocab size: {len(vocab):,}")

    X_train = texts_to_sequences(train_texts, vocab, MAX_LEN)
    X_val   = texts_to_sequences(val_texts,   vocab, MAX_LEN)

    # ── GloVe ─────────────────────────────────────────────────────────────────
    embed_matrix = load_glove(GLOVE_PATH, vocab, EMBED_DIM)

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    train_ds = TextDataset(X_train, Y_train)
    val_ds   = TextDataset(X_val,   Y_val)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 4, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BiLSTMClassifier(
        vocab_size   = len(vocab),
        embed_dim    = EMBED_DIM,
        embed_matrix = embed_matrix,
        hidden_dim   = HIDDEN_DIM,
        num_layers   = NUM_LAYERS,
        num_labels   = num_labels,
        dropout      = DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Loss: BCEWithLogitsLoss with per-label pos_weight ─────────────────────
    # Up-weight rare labels; cap to avoid gradient explosions on 1-example labels
    pos_counts = Y_train.sum(axis=0).clip(min=1)
    neg_counts = len(Y_train) - pos_counts
    raw_pw     = neg_counts / pos_counts
    pos_weight = torch.tensor(
        np.clip(raw_pw, None, POS_WEIGHT_CAP), dtype=torch.float32
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_micro   = -1.0
    best_thr     = 0.5
    patience_ctr = 0

    print(f"\n{'─'*65}")
    print(f"  Epoch  |  train loss  |  val Micro F1  |  threshold")
    print(f"{'─'*65}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0

        for X, Y in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss   = criterion(logits, Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches

        val_probs, val_labels = get_val_probs(model, val_loader, DEVICE)
        thr, micro = sweep_threshold(val_probs, val_labels)

        improved = micro > best_micro + 1e-5
        marker   = " ★" if improved else ""
        print(f"  {epoch:4d}   |  {avg_loss:.4f}      |  {micro:.4f}         |  {thr:.2f}{marker}")

        if improved:
            best_micro   = micro
            best_thr     = thr
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "bilstm.pt"))
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"{'─'*65}")
    print(f"\nBest val Micro F1: {best_micro:.4f}  @ threshold={best_thr:.2f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    with open(os.path.join(MODEL_DIR, "mlb.pkl"), "wb") as f:
        pickle.dump(mlb, f)

    meta = {
        "best_threshold": best_thr,
        "num_labels":     num_labels,
        "max_len":        MAX_LEN,
        "embed_dim":      EMBED_DIM,
        "hidden_dim":     HIDDEN_DIM,
        "num_layers":     NUM_LAYERS,
        "dropout":        DROPOUT,
        "vocab_size":     len(vocab),
        "val_micro_f1":   round(best_micro, 4),
        "labels":         labels,
    }
    with open(os.path.join(MODEL_DIR, "meta_bilstm.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nArtifacts saved:")
    for fname in ("bilstm.pt", "vocab.pkl", "mlb.pkl", "meta_bilstm.json"):
        fpath = os.path.join(MODEL_DIR, fname)
        size  = os.path.getsize(fpath) / 1e6
        print(f"  {fpath}  ({size:.1f} MB)")

    if best_micro < 0.82:
        print("\n[HINT] val Micro F1 < 0.82 — try setting HIDDEN_DIM = 512 and re-running.")


if __name__ == "__main__":
    main()
