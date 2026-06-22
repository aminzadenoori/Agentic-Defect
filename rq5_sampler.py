"""
rq5_sampler.py — Build the human-study CSV from LLM-vs-gold disagreements.

For RQ5 we want a stratified sample of requirements where the LLMs collectively
disagree with the published QuRE label. The output CSV is what 2-3 human raters
will adjudicate (blind to which label is gold).

Strategy:
  1. Across all (method, model, fold) predictions, count for each requirement
     how many configurations predicted `defect` vs. `ok`.
  2. "Contested" requirements are those where the cross-configuration majority
     disagrees with the gold label.
  3. Stratify the sample on:
       - disagreement direction (LLM says ok / gold defect, vs. opposite)
       - the dominant method behind the LLM majority (proportional)
       - model family (proportional)
  4. For each sampled requirement, include up to N=3 reasoning traces from the
     configurations that drove the majority. The human rater sees the requirement
     + reasoning, NOT the gold label.

Output CSV columns:
    sample_id, requirement_id, requirement_text, llm_majority_label,
    gold_label, n_configs, n_agree_with_llm, n_agree_with_gold,
    reasoning_1, method_1, model_1,
    reasoning_2, method_2, model_2,
    reasoning_3, method_3, model_3,
    human_rater_1_verdict, human_rater_1_confidence, human_rater_1_notes,
    (same for raters 2, 3)

Usage:
    python rq5_sampler.py --in results/ --out paper/human_study/sample.csv --n 150
"""

from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import load_all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reasonings_per_case", type=int, default=3)
    args = ap.parse_args()

    df = load_all_results(Path(args.in_dir))
    rng = np.random.default_rng(args.seed)

    # 1) per requirement_id, count configurations and their predictions
    grouped = df.groupby("requirement_id")
    candidates = []
    for rid, g in grouped:
        n_configs = len(g)
        if n_configs < 4:  # skip requirements with too little signal
            continue
        votes = Counter(g["pred_label"])
        if "ok" not in votes: votes["ok"] = 0
        if "defect" not in votes: votes["defect"] = 0
        llm_majority = "defect" if votes["defect"] > votes["ok"] else "ok"
        gold = g["gold_label"].iloc[0]
        if llm_majority == gold:
            continue  # not a disagreement
        # supporting configurations (those that voted with the LLM majority)
        supporters = g[g["pred_label"] == llm_majority].copy()
        # add diversity: rank by model+method, take up to N
        supporters = supporters.drop_duplicates(subset=["model", "method"])
        candidates.append({
            "requirement_id": int(rid),
            "requirement_text": g["requirement_text"].iloc[0],
            "weak_words_annotated": g["weak_words_annotated"].iloc[0],
            "llm_majority_label": llm_majority,
            "gold_label": gold,
            "n_configs": int(n_configs),
            "n_agree_with_llm": int(votes[llm_majority]),
            "n_agree_with_gold": int(votes[gold]),
            "_supporters": supporters,
        })

    if not candidates:
        raise SystemExit("No disagreements found.")

    cdf = pd.DataFrame([{k: v for k, v in c.items() if k != "_supporters"} for c in candidates])
    print(f"[mined] {len(cdf)} requirements with cross-config majority ≠ gold")

    # 2) stratified sampling by direction × dominant method × model family
    cdf["direction"] = cdf["llm_majority_label"] + "_vs_gold_" + cdf["gold_label"]
    # we don't have family directly, but the model slug encodes it
    def _family(m): return m.split("-")[0]
    # pull dominant method+model from supporters for stratification
    for i, row in cdf.iterrows():
        sup = candidates[i]["_supporters"]
        cdf.at[i, "dominant_method"] = sup["method"].value_counts().idxmax()
        cdf.at[i, "dominant_model"] = sup["model"].value_counts().idxmax()
    cdf["dominant_family"] = cdf["dominant_model"].map(_family)

    # proportional sampling: stratify on direction × family
    strat = cdf["direction"] + "::" + cdf["dominant_family"]
    counts = strat.value_counts(normalize=True)
    target_per = (counts * args.n).round().astype(int)
    sampled_ids = []
    for stratum, k in target_per.items():
        pool = cdf[strat == stratum]
        if len(pool) == 0 or k == 0: continue
        take = min(k, len(pool))
        idx = rng.choice(pool.index, size=take, replace=False)
        sampled_ids.extend(idx.tolist())

    # top up to args.n if rounding lost some
    remaining = args.n - len(sampled_ids)
    if remaining > 0:
        leftover = cdf.index.difference(sampled_ids)
        sampled_ids.extend(rng.choice(leftover, size=min(remaining, len(leftover)), replace=False).tolist())
    sampled = cdf.loc[sampled_ids].reset_index(drop=True)
    print(f"[sampled] n={len(sampled)} cases")

    # 3) attach reasoning traces and write CSV
    out_rows = []
    for sid, row in sampled.iterrows():
        rid = int(row["requirement_id"])
        sup = candidates[[c["requirement_id"] for c in candidates].index(rid)]["_supporters"]
        # diversity: prefer different method+model triples
        sup = sup.drop_duplicates(subset=["model", "method"]).head(args.reasonings_per_case)
        rec = {
            "sample_id": sid,
            "requirement_id": rid,
            "requirement_text": row["requirement_text"],
            "weak_words_annotated": row["weak_words_annotated"],
            "llm_majority_label": row["llm_majority_label"],
            "gold_label": row["gold_label"],     # remove before sending to raters!
            "n_configs": row["n_configs"],
            "n_agree_with_llm": row["n_agree_with_llm"],
            "n_agree_with_gold": row["n_agree_with_gold"],
        }
        for i, (_, srow) in enumerate(sup.iterrows(), start=1):
            rec[f"reasoning_{i}"] = srow["reasoning"]
            rec[f"method_{i}"]    = srow["method"]
            rec[f"model_{i}"]     = srow["model"]
        for i in range(len(sup) + 1, args.reasonings_per_case + 1):
            rec[f"reasoning_{i}"] = ""
            rec[f"method_{i}"]    = ""
            rec[f"model_{i}"]     = ""
        # rater columns (empty for now)
        for r in (1, 2, 3):
            rec[f"human_rater_{r}_verdict"] = ""
            rec[f"human_rater_{r}_confidence"] = ""
            rec[f"human_rater_{r}_notes"] = ""
        out_rows.append(rec)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")
    print("\n>>> IMPORTANT: before sending to human raters, drop the `gold_label` column")
    print(">>> (and ideally also the `llm_majority_label`) to keep the study blind.")


if __name__ == "__main__":
    main()
