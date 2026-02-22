"""
preprocess.py – EDA for Track 2 multi-label text classification
Run from: datathon-26/src/track2/
  python ../preprocess.py
Outputs: prints stats to stdout + saves plots to eda_plots/
"""

import os
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for WSL / headless)
import matplotlib.pyplot as plt
import seaborn as sns

# ── paths (relative to src/track2/) ──────────────────────────────────────────
TRAIN_PATH      = "train.csv"
VAL_PATH        = "val.csv"
LABEL_LIST_PATH = "label_list.txt"
PLOT_DIR        = "eda_plots"

os.makedirs(PLOT_DIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def parse_topics(x: str) -> list[str]:
    x = (x or "none").strip().lower()
    parts = [t.strip() for t in x.split("|") if t.strip()]
    return parts if parts else ["none"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(TRAIN_PATH, dtype=str).fillna("")
    val   = pd.read_csv(VAL_PATH,   dtype=str).fillna("")
    labels = [l.strip().lower() for l in open(LABEL_LIST_PATH) if l.strip()]
    train["label_list"] = train["topics"].apply(parse_topics)
    val["label_list"]   = val["topics"].apply(parse_topics)
    train["full_text"]  = (train["title"] + " " + train["text"]).str.strip()
    val["full_text"]    = (val["title"]   + " " + val["text"]).str.strip()
    return train, val, labels


def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ── 1. Basic dataset stats ────────────────────────────────────────────────────
def basic_stats(train: pd.DataFrame, val: pd.DataFrame) -> None:
    section("1. BASIC DATASET STATS")
    for name, df in [("TRAIN", train), ("VAL", val)]:
        n_labels_per_doc = df["label_list"].apply(len)
        print(f"\n{name}: {len(df):,} articles")
        print(f"  Labels per article  mean={n_labels_per_doc.mean():.2f}  "
              f"median={n_labels_per_doc.median():.0f}  "
              f"max={n_labels_per_doc.max()}  "
              f"min={n_labels_per_doc.min()}")
        vc = n_labels_per_doc.value_counts().sort_index()
        print("  Distribution of #labels per article:")
        for k, v in vc.items():
            bar = "█" * min(40, int(40 * v / len(df)))
            print(f"    {k:3d} label(s): {v:6,} ({100*v/len(df):5.1f}%) {bar}")


# ── 2. Class imbalance ────────────────────────────────────────────────────────
def class_imbalance(train: pd.DataFrame, val: pd.DataFrame, labels: list[str]) -> None:
    section("2. CLASS IMBALANCE")

    all_labels = [l for doc in train["label_list"] for l in doc]
    freq = Counter(all_labels)

    # align to official label list order
    counts = pd.Series({l: freq.get(l, 0) for l in labels}, name="count")
    counts = counts.sort_values(ascending=False)

    total = len(train)
    print(f"\nTotal label occurrences in TRAIN: {sum(freq.values()):,}")
    print(f"Documents with 'none' label: {freq.get('none', 0):,}")
    print(f"\nTop-20 labels (by frequency):")
    print(f"{'Rank':<5} {'Label':<22} {'Count':>7}  {'% docs':>7}  Bar")
    for rank, (label, cnt) in enumerate(counts.head(20).items(), 1):
        pct = 100 * cnt / total
        bar = "█" * min(50, int(50 * cnt / counts.iloc[0]))
        print(f"{rank:<5} {label:<22} {cnt:>7,}  {pct:>6.1f}%  {bar}")

    print(f"\nBottom-20 labels (by frequency):")
    print(f"{'Label':<22} {'Count':>7}  {'% docs':>7}")
    for label, cnt in counts.tail(20).iloc[::-1].items():
        pct = 100 * cnt / total
        print(f"{label:<22} {cnt:>7,}  {pct:>6.1f}%")

    # imbalance ratio (excluding 'none')
    active = counts.drop("none", errors="ignore")
    ratio = active.iloc[0] / max(active.iloc[-1], 1)
    print(f"\nImbalance ratio (most / least frequent active label): {ratio:.1f}x")
    print(f"  Most frequent : {active.index[0]} ({active.iloc[0]:,})")
    print(f"  Least frequent: {active.index[-1]} ({active.iloc[-1]:,})")

    # ── plot: top-40 label frequency bar chart ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    top40 = counts.head(40)
    ax.bar(range(len(top40)), top40.values, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(top40)))
    ax.set_xticklabels(top40.index, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Frequency (train)")
    ax.set_title("Top-40 Label Frequencies (Train Set)")
    ax.set_yscale("log")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "label_freq_top40.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {path}")

    # ── plot: full label frequency sorted (log scale) ─────────────────────
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(range(len(counts)), counts.values, color="darkorange", width=1.0)
    ax.set_xlabel("Label rank")
    ax.set_ylabel("Frequency (log)")
    ax.set_yscale("log")
    ax.set_title("All-label Frequency Distribution (Train, log scale)")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "label_freq_all.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")

    # ── val vs train label balance ────────────────────────────────────────
    val_freq = Counter([l for doc in val["label_list"] for l in doc])
    val_counts = pd.Series({l: val_freq.get(l, 0) for l in labels})
    val_counts = val_counts.reindex(counts.index)  # same order as train

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (name, cnts) in zip(axes, [("Train", counts), ("Val", val_counts)]):
        top = cnts.head(30)
        ax.barh(range(len(top)), top.values[::-1], color="teal")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index[::-1], fontsize=7)
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top-30 Labels – {name}")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "label_freq_train_vs_val.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


# ── 3. Label co-occurrence ────────────────────────────────────────────────────
def label_cooccurrence(train: pd.DataFrame, labels: list[str]) -> None:
    section("3. LABEL CO-OCCURRENCE")

    # Build co-occurrence matrix
    label_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    co = np.zeros((n, n), dtype=np.int32)

    for label_set in train["label_list"]:
        idxs = [label_idx[l] for l in label_set if l in label_idx]
        for i in idxs:
            for j in idxs:
                co[i, j] += 1

    co_df = pd.DataFrame(co, index=labels, columns=labels)

    # Top co-occurring PAIRS (off-diagonal)
    pairs = []
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if j <= i:
                continue
            pairs.append((la, lb, int(co[i, j])))

    pairs.sort(key=lambda x: -x[2])
    print(f"\nTop-30 co-occurring label pairs (in TRAIN):")
    print(f"{'Label A':<22} {'Label B':<22} {'Co-occur':>9}  Bar")
    for la, lb, cnt in pairs[:30]:
        bar = "█" * min(40, int(40 * cnt / pairs[0][2]))
        print(f"{la:<22} {lb:<22} {cnt:>9,}  {bar}")

    # Zero co-occurrence report
    zero_pairs = [(la, lb) for la, lb, cnt in pairs if cnt == 0]
    print(f"\nLabel pairs that NEVER co-occur: {len(zero_pairs):,} / {len(pairs):,} total pairs")

    # Average number of co-occurring labels per label
    print(f"\nAverage co-occurrence partners per label (excl. self):")
    top_labels = sorted(
        [(l, int(co_df.loc[l].drop(l, errors="ignore").sum()))
         for l in labels if l != "none"],
        key=lambda x: -x[1]
    )
    print(f"{'Label':<22} {'Total co-occur':>15}")
    for label, total in top_labels[:20]:
        print(f"  {label:<22} {total:>15,}")

    # ── plot: co-occurrence heatmap (top-30 active labels only) ──────────
    # pick top-30 by marginal count (excluding 'none')
    active_counts = {l: int(co_df.loc[l, l]) for l in labels if l != "none"}
    top30 = sorted(active_counts, key=lambda x: -active_counts[x])[:30]
    sub = co_df.loc[top30, top30].values.copy().astype(float)
    np.fill_diagonal(sub, 0)  # blank diagonal to focus on pairs
    sub = pd.DataFrame(sub, index=top30, columns=top30)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        sub,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="white",
        annot=False,
        fmt="d",
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title("Label Co-occurrence Heatmap (Top-30 by frequency, diagonal=0)", fontsize=11)
    ax.tick_params(axis="x", labelsize=7, rotation=60)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "label_cooccurrence_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {path}")

    # ── plot: bar chart of top-20 co-occurring pairs ──────────────────────
    top_pairs = pairs[:20]
    pair_labels = [f"{a}+{b}" for a, b, _ in top_pairs]
    pair_counts = [c for _, _, c in top_pairs]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(range(len(pair_labels)), pair_counts[::-1], color="mediumseagreen")
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels[::-1], fontsize=8)
    ax.set_xlabel("Co-occurrence count (Train)")
    ax.set_title("Top-20 Label Co-occurrence Pairs")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "label_cooccurrence_top20_pairs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


# ── 4. Text length analysis ───────────────────────────────────────────────────
def text_lengths(train: pd.DataFrame, val: pd.DataFrame) -> None:
    section("4. TEXT LENGTH ANALYSIS")

    percentiles = [5, 25, 50, 75, 90, 95, 99]

    for name, df in [("TRAIN", train), ("VAL", val)]:
        print(f"\n── {name} ──")
        for col, label in [("title", "Title"), ("text", "Body"), ("full_text", "Title+Body")]:
            words  = df[col].str.split().str.len().fillna(0).astype(int)
            chars  = df[col].str.len().fillna(0).astype(int)
            print(f"\n  {label}:")
            print(f"    Words  → mean={words.mean():.1f}  median={words.median():.0f}  "
                  f"min={words.min()}  max={words.max():,}")
            print(f"    Chars  → mean={chars.mean():.1f}  median={chars.median():.0f}  "
                  f"min={chars.min()}  max={chars.max():,}")
            pcts_w = np.percentile(words, percentiles)
            pcts_c = np.percentile(chars, percentiles)
            print(f"    Word percentiles  ({percentiles}): {[int(p) for p in pcts_w]}")
            print(f"    Char percentiles  ({percentiles}): {[int(p) for p in pcts_c]}")

    # ── plot: word-count distributions (body text, train vs val) ─────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    train_words = train["text"].str.split().str.len().fillna(0).astype(int)
    val_words   = val["text"].str.split().str.len().fillna(0).astype(int)

    cap = int(np.percentile(train_words, 99))  # clip to 99th pct for readability
    for ax, (name, words) in zip(axes, [("Train", train_words), ("Val", val_words)]):
        clipped = words.clip(upper=cap)
        ax.hist(clipped, bins=60, color="royalblue", edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.axvline(words.median(), color="red",    linestyle="--", linewidth=1.5, label=f"median={words.median():.0f}")
        ax.axvline(words.mean(),   color="orange", linestyle="--", linewidth=1.5, label=f"mean={words.mean():.0f}")
        ax.set_xlabel("Word count (body, clipped at 99th pct)")
        ax.set_ylabel("# articles")
        ax.set_title(f"Body Word-Count Distribution – {name}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "text_word_count_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {path}")

    # ── plot: word-count box plots by #labels per article ────────────────
    train_copy = train.copy()
    train_copy["n_labels"] = train_copy["label_list"].apply(len).clip(upper=5)
    train_copy["word_count"] = train_copy["text"].str.split().str.len().fillna(0)
    grouped = [
        train_copy.loc[train_copy["n_labels"] == k, "word_count"].clip(upper=cap).values
        for k in sorted(train_copy["n_labels"].unique())
    ]
    group_labels = [f"{k}" if k < 5 else "5+" for k in sorted(train_copy["n_labels"].unique())]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(grouped, labels=group_labels, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_xlabel("Number of labels per article")
    ax.set_ylabel("Word count (body)")
    ax.set_title("Body Word Count vs. Number of Labels (Train)")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "word_count_by_num_labels.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")

    # ── plot: title word-count distribution ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    train_title_words = train["title"].str.split().str.len().fillna(0).astype(int)
    ax.hist(train_title_words, bins=40, color="mediumpurple", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(train_title_words.median(), color="red",    linestyle="--", linewidth=1.5,
               label=f"median={train_title_words.median():.0f}")
    ax.axvline(train_title_words.mean(),   color="orange", linestyle="--", linewidth=1.5,
               label=f"mean={train_title_words.mean():.1f}")
    ax.set_xlabel("Word count (title)")
    ax.set_ylabel("# articles")
    ax.set_title("Title Word-Count Distribution – Train")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "title_word_count_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    train, val, labels = load_data()
    print(f"  train: {len(train):,} rows  |  val: {len(val):,} rows  |  labels: {len(labels)}")

    basic_stats(train, val)
    class_imbalance(train, val, labels)
    label_cooccurrence(train, labels)
    text_lengths(train, val)

    section("DONE")
    print(f"All plots saved to: {os.path.abspath(PLOT_DIR)}/\n")


if __name__ == "__main__":
    main()
