"""
metrics.py — Aggregate per-prediction CSVs into the paper's tables.

Reads results/*.csv (one per method × model × fold), computes:
  - F1-macro mean ± std across folds (with bootstrap 95% CI)
  - Accuracy, per-class F1
  - Per-weak-word F1 (using the annotated weak words)
  - McNemar's significance test between MARQ and each baseline (paired by requirement)
  - Cost per prediction: tokens, latency, n_llm_calls

Outputs paper-ready CSV tables to <out>/.

Usage:
    python metrics.py compile --in results/ --out paper/tables/
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


LABELS = ["ok", "defect"]


def _bootstrap_f1(y_true, y_pred, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    N = len(y_true)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        samples.append(f1_score(y_true[idx], y_pred[idx],
                                labels=LABELS, average="macro", zero_division=0))
    return float(np.mean(samples)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _mcnemar(y_true, p_a, p_b) -> dict:
    """McNemar's test on paired predictions."""
    y_true = np.array(y_true); p_a = np.array(p_a); p_b = np.array(p_b)
    a_correct = (p_a == y_true); b_correct = (p_b == y_true)
    n01 = int(((~a_correct) & b_correct).sum())   # A wrong, B right
    n10 = int((a_correct & (~b_correct)).sum())   # A right, B wrong
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "stat": 0.0, "p_value": 1.0}
    # continuity-corrected chi-squared
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    from scipy.stats import chi2
    p = 1.0 - chi2.cdf(stat, df=1)
    return {"n01": n01, "n10": n10, "stat": float(stat), "p_value": float(p)}


def load_all_results(results_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(results_dir.glob("*.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")
    if not frames:
        raise SystemExit(f"No CSVs found in {results_dir}")
    df = pd.concat(frames, ignore_index=True)
    # normalize
    df["pred_label"] = df["pred_label"].astype(str).str.lower().str.strip()
    df["gold_label"] = df["gold_label"].astype(str).str.lower().str.strip()
    return df


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """F1-macro mean ± std across folds, per (method, model)."""
    rows = []
    for (method, model), g in df.groupby(["method", "model"]):
        fold_f1s = []
        for fold, gf in g.groupby("fold"):
            fold_f1s.append(f1_score(gf["gold_label"], gf["pred_label"],
                                     labels=LABELS, average="macro", zero_division=0))
        if not fold_f1s: continue
        mean_f1, lo, hi = _bootstrap_f1(g["gold_label"].values, g["pred_label"].values)
        rows.append({
            "method": method,
            "model": model,
            "n_folds": len(fold_f1s),
            "f1_macro_mean": round(float(np.mean(fold_f1s)), 4),
            "f1_macro_std": round(float(np.std(fold_f1s)), 4),
            "f1_macro_boot_lo": round(lo, 4),
            "f1_macro_boot_hi": round(hi, 4),
            "accuracy": round(float((g["pred_label"] == g["gold_label"]).mean()), 4),
            "n_predictions": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values(["model", "method"]).reset_index(drop=True)


def per_weak_word_table(df: pd.DataFrame) -> pd.DataFrame:
    """Break down F1 by the weak word(s) annotated in each requirement.

    A requirement contributes to a weak word's bucket if that word was annotated for it.
    A requirement may contribute to multiple buckets.
    """
    rows = []
    for (method, model), g in df.groupby(["method", "model"]):
        buckets = defaultdict(list)  # weak_word -> list[(gold, pred)]
        for _, r in g.iterrows():
            try:
                words = json.loads(r["weak_words_annotated"])
            except Exception:
                continue
            for w in words:
                buckets[w].append((r["gold_label"], r["pred_label"]))
        for w, pairs in buckets.items():
            if len(pairs) < 5: continue
            y, p = zip(*pairs)
            if len(set(y)) < 2:
                rows.append({"method": method, "model": model, "weak_word": w,
                             "n": len(pairs), "f1_macro": None,
                             "acc": float(np.mean(np.array(p) == np.array(y))),
                             "note": "single-class"})
            else:
                rows.append({"method": method, "model": model, "weak_word": w,
                             "n": len(pairs),
                             "f1_macro": round(f1_score(y, p, labels=LABELS,
                                                       average="macro", zero_division=0), 4),
                             "acc": round(float(np.mean(np.array(p) == np.array(y))), 4)})
    return pd.DataFrame(rows).sort_values(["model", "method", "weak_word"]).reset_index(drop=True)


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-prediction cost: tokens, latency, n_llm_calls. Mean across all predictions."""
    rows = []
    for (method, model), g in df.groupby(["method", "model"]):
        rows.append({
            "method": method,
            "model": model,
            "mean_llm_calls": round(float(g["n_llm_calls"].mean()), 2),
            "mean_input_tokens": round(float(g["input_tokens"].mean()), 1),
            "mean_output_tokens": round(float(g["output_tokens"].mean()), 1),
            "mean_total_tokens": round(float((g["input_tokens"] + g["output_tokens"]).mean()), 1),
            "mean_latency_ms": round(float(g["latency_ms"].mean()), 1),
        })
    return pd.DataFrame(rows).sort_values(["model", "method"]).reset_index(drop=True)


def mcnemar_table(df: pd.DataFrame, target_method: str = "marq") -> pd.DataFrame:
    """Pairwise McNemar between target_method and each other method, per model."""
    rows = []
    for model, g in df.groupby("model"):
        target = g[g["method"] == target_method].set_index(["fold", "requirement_id"])
        for method in g["method"].unique():
            if method == target_method: continue
            other = g[g["method"] == method].set_index(["fold", "requirement_id"])
            joined = target.join(other, lsuffix="_t", rsuffix="_o", how="inner")
            if len(joined) == 0: continue
            r = _mcnemar(joined["gold_label_t"], joined["pred_label_t"], joined["pred_label_o"])
            rows.append({
                "model": model,
                f"{target_method}_vs": method,
                "n_paired": len(joined),
                "n01_target_only_correct": r["n01"],
                "n10_other_only_correct": r["n10"],
                "chi2_stat": round(r["stat"], 3),
                "p_value": round(r["p_value"], 6),
                "favors": (target_method if r["n01"] > r["n10"] else method) if r["p_value"] < 0.05 else "n.s.",
            })
    return pd.DataFrame(rows)


def compile_cmd(args):
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(in_dir)
    print(f"[loaded] {len(df)} predictions across "
          f"{df['method'].nunique()} methods × {df['model'].nunique()} models × {df['fold'].nunique()} folds")

    headline = headline_table(df)
    headline.to_csv(out_dir / "table_headline.csv", index=False)
    print(f"[saved] {out_dir / 'table_headline.csv'}")
    print(headline.to_string(index=False))

    pw = per_weak_word_table(df)
    pw.to_csv(out_dir / "table_per_weak_word.csv", index=False)
    print(f"\n[saved] {out_dir / 'table_per_weak_word.csv'} ({len(pw)} rows)")

    cost = cost_table(df)
    cost.to_csv(out_dir / "table_cost.csv", index=False)
    print(f"\n[saved] {out_dir / 'table_cost.csv'}")
    print(cost.to_string(index=False))

    mc = mcnemar_table(df, target_method="marq")
    mc.to_csv(out_dir / "table_mcnemar.csv", index=False)
    print(f"\n[saved] {out_dir / 'table_mcnemar.csv'}")
    if len(mc): print(mc.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--in", dest="in_dir", required=True)
    c.add_argument("--out", dest="out_dir", required=True)
    args = ap.parse_args()
    if args.cmd == "compile":
        compile_cmd(args)


if __name__ == "__main__":
    main()
