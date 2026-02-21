# Track 2 – Text Classification

## Task
Multi-label text classification on a collection of newswire articles. Each
article may belong to one or more topic categories. Organizers will run your
`predict.py` against the withheld test set after submission.

---

## Files Provided

| File | Description |
|------|-------------|
| `train.csv` | Training data with labels |
| `val.csv` | Validation data with labels — use to self-evaluate |
| `label_list.txt` | All valid topic labels |
| `predict.py` | Inference template — implement this and submit it |

---

## Categories

**Category 1 – With Pretrained Embeddings**
Pretrained word embeddings are allowed.
These may be used as fixed features or fine-tuned. No large language models.

**Category 2 – Without Pretrained Embeddings**
All representations must be learned from scratch using the provided data only.
Allowed approaches include bag-of-words, TF-IDF, n-gram features, or randomly
initialized embeddings.

---

## Label Format
- Topics are pipe-separated: `earn|trade`, `money-fx`, `none`
- Articles with no predicted label must use `none`
- Label order does not matter — `earn|trade` equals `trade|earn`
- Labels are case-sensitive — use lowercase exactly as in `label_list.txt`

---

## How to Self-Evaluate
Set the paths at the top of `predict.py` to point at the val set:

```python
INPUT_PATH  = "val.csv"
OUTPUT_PATH = "val_predictions.csv"
LABELS_PATH = "val.csv"
MODEL_PATH  = "model/"
```

Then run:
```bash
python predict.py
```

This prints your Micro F1, Macro F1, exact-match accuracy, and Hamming
accuracy. Your final leaderboard ranking is based on **Micro F1** on the
hidden test set evaluated by organizers.

---

## What to Submit
A single zip file named `teamname_track2_cat[1or2].zip`:

```
teamname_track2_cat[1or2].zip
├── predict.py        ← paths must be set to test before submitting
├── train.py          ← your training script
├── model/            ← saved model weights
├── requirements.txt  ← all dependencies with pinned versions
└── report.pdf        ← technical description (max 2 pages)
```

### `predict.py` Requirements
Before submitting, ensure paths at the top of `predict.py` are:

```python
INPUT_PATH  = "test.csv"
OUTPUT_PATH = "predictions.csv"
LABELS_PATH = None
MODEL_PATH  = "model/"
```

Organizers will place your `predict.py` and `model/` alongside the withheld
test set and run:
```bash
pip install -r requirements.txt
python predict.py
```

**Your `predict.py` must:**
- Load weights from `MODEL_PATH`
- Return one pipe-separated prediction string per article
- Run without internet access
- Complete in under 10 minutes
- Not retrain the model — inference only

### `report.pdf` (max 2 pages)
- Preprocessing steps
- Model architecture
- Training procedure and hyperparameters
- Explicit statement of category compliance (Category 1 or 2)

---

## Evaluation Metric
Submissions are ranked by **Micro F1 score** on the hidden test set.

Micro F1 aggregates true positives, false positives, and false negatives
across all labels before computing precision and recall. It handles class
imbalance better than raw accuracy for imbalanced multi-label datasets.

---

## Rules
- No large language models in any form
- Only the provided dataset files may be used — no external data
- **Category 1:** pretrained word embeddings allowed
- **Category 2:** all representations must be learned from scratch on provided data only
- Submissions must be fully reproducible
- Non-reproducible submissions will be disqualified