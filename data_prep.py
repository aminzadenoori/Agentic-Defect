"""
data_prep.py — QuRE → requirement-level + 5-fold stratified 30/70 splits.

Pipeline:
  1. Load QuRE.csv (rows are (requirement, weak_word, defect)).
  2. Dedupe and drop degenerate weak words.
  3. Aggregate to requirement level — a requirement is `defect` if ANY of its
     weak-word instances is `defect`, otherwise `ok`. Per-word annotations kept
     as a JSON metadata column.
  4. Generate N stratified 30/70 splits (StratifiedShuffleSplit) on the
     requirement-level label.

Output: splits/fold_{k}/pool.parquet, splits/fold_{k}/test.parquet for k in [0..N-1]
Plus splits/meta.json with split statistics.

Usage:
    python data_prep.py --csv QuRE.csv --out splits/ --n_folds 5 --seed 42
"""

from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


DEGENERATE_WORDS = {"common", "comparable", "similar"}


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["requirement"] = df["requirement"].astype(str).str.strip()
    df["weak_word"] = df["weak_word"].astype(str).str.strip().str.lower()
    df["defect"] = df["defect"].astype(str).str.strip().str.lower()
    df = df[(df["requirement"] != "") & df["defect"].isin({"ok", "defect"})]
    df = df.drop_duplicates(subset=["requirement", "weak_word"]).reset_index(drop=True)
    df = df[~df["weak_word"].isin(DEGENERATE_WORDS)].reset_index(drop=True)
    return df


def aggregate_to_requirement_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique requirement.

    Columns:
      requirement_id, requirement, weak_words (list[str]),
      per_word_labels (dict[str,str]), label (ok|defect).
    """
    grouped = df.groupby("requirement")
    rows = []
    for rid, (req, sub) in enumerate(grouped):
        weak_words = sub["weak_word"].tolist()
        per_word = dict(zip(sub["weak_word"], sub["defect"]))
        # any-defect aggregation
        label = "defect" if "defect" in per_word.values() else "ok"
        rows.append({
            "requirement_id": rid,
            "requirement": req,
            "weak_words": weak_words,
            "per_word_labels": per_word,
            "label": label,
        })
    out = pd.DataFrame(rows)
    return out


def make_folds(df: pd.DataFrame, n_folds: int, seed: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """5 stratified 30/70 splits. Each call returns (pool_df, test_df)."""
    splitter = StratifiedShuffleSplit(n_splits=n_folds, train_size=0.30, random_state=seed)
    out = []
    for pool_idx, test_idx in splitter.split(df, df["label"]):
        pool = df.iloc[pool_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        out.append((pool, test))
    return out


def write_splits(folds, out_dir: Path) -> dict:
    """Write each fold to parquet; return summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"n_folds": len(folds), "folds": []}
    for k, (pool, test) in enumerate(folds):
        fdir = out_dir / f"fold_{k}"
        fdir.mkdir(exist_ok=True)
        pool.to_parquet(fdir / "pool.parquet", index=False)
        test.to_parquet(fdir / "test.parquet", index=False)
        summary["folds"].append({
            "fold": k,
            "pool_n": len(pool),
            "test_n": len(test),
            "pool_labels": Counter(pool["label"]),
            "test_labels": Counter(test["label"]),
        })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="QuRE.csv path")
    ap.add_argument("--out", default="splits/", help="output directory")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = load_and_clean(args.csv)
    print(f"[load] {len(raw)} (requirement, weak_word) rows after dedup")

    req_level = aggregate_to_requirement_level(raw)
    label_counts = Counter(req_level["label"])
    print(f"[aggregate] {len(req_level)} unique requirements  labels={dict(label_counts)}")

    folds = make_folds(req_level, n_folds=args.n_folds, seed=args.seed)

    out = Path(args.out)
    summary = write_splits(folds, out)
    summary["total_requirements"] = len(req_level)
    summary["overall_label_distribution"] = dict(label_counts)
    summary["seed"] = args.seed

    with open(out / "meta.json", "w") as f:
        # Counters → plain dicts for JSON
        for entry in summary["folds"]:
            entry["pool_labels"] = dict(entry["pool_labels"])
            entry["test_labels"] = dict(entry["test_labels"])
        json.dump(summary, f, indent=2)

    print(f"\n[done] wrote {args.n_folds} folds to {out}/")
    for entry in summary["folds"]:
        print(f"  fold {entry['fold']}: pool n={entry['pool_n']:4d}  test n={entry['test_n']:4d}  "
              f"pool {entry['pool_labels']}  test {entry['test_labels']}")


if __name__ == "__main__":
    main()
