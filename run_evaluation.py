"""
run_evaluation.py — Main evaluation grid: methods × models × folds.

Outputs one CSV per (method, model, fold) at:
    results/{method}__{model_slug}__fold{k}.csv

Each row contains the full prediction record: gold label, predicted label, full
reasoning string, token counts, latency, retrieved pool IDs. This is the central
artifact — both for metrics aggregation and for RQ5's human study.

Resumable: skips any (method, model, fold) whose CSV already exists.
Quick smoke test: --quick uses only fold 0, the smallest model, and 30 test rows.

Usage:
    python run_evaluation.py --splits splits/ --out results/
    python run_evaluation.py --splits splits/ --out results/ --quick
    python run_evaluation.py --splits splits/ --out results/ \\
        --methods zero_shot few_shot cot marq \\
        --models qwen-1.5b qwen-7b llama-3b gemma-2b phi-3.5 \\
        --folds 0 1 2 3 4
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from baselines import (
    LLMBackend, run_zero_shot, run_few_shot, run_cot,
)
from orchestrator import run_marq
from retrieval import PoolRetriever
from huggingface_hub import login

# Login to Hugging Face (DO NOT CHANGE TOKEN)
login(token="meta-llama/Llama-3.1-8B-Instruct")


MODEL_REGISTRY = {
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-7b":   "Qwen/Qwen2.5-7B-Instruct",
    "llama-3b":  "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-2b":  "google/gemma-2-2b-it",
    "phi-3.5":   "microsoft/Phi-3.5-mini-instruct",
}

METHODS = ["zero_shot", "few_shot", "cot", "marq"]


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def load_fold(splits_dir: Path, k: int):
    fdir = splits_dir / f"fold_{k}"
    pool = pd.read_parquet(fdir / "pool.parquet")
    test = pd.read_parquet(fdir / "test.parquet")
    cot_path = fdir / "cot_chains.json"
    cot_cache: Dict[int, str] = {}
    if cot_path.exists():
        cot_cache = {int(k_): v for k_, v in json.loads(cot_path.read_text()).items()}
    return pool, test, cot_cache


def run_one_method_one_fold(method: str, model_name: str, model_slug: str, fold: int,
                            test: pd.DataFrame, retriever: PoolRetriever | None,
                            cot_cache: Dict[int, str], out_dir: Path,
                            k: int = 5) -> Path:
    out_path = out_dir / f"{method}__{model_slug}__fold{fold}.csv"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return out_path

    llm = LLMBackend.get(model_name)
    records: List[Dict] = []
    for _, row in tqdm(test.iterrows(), total=len(test), desc=f"{method}/{model_slug}/f{fold}"):
        req = row["requirement"]
        t0 = time.time()
        if method == "zero_shot":
            r = run_zero_shot(llm, req)
        elif method == "few_shot":
            demos = retriever.retrieve(req, k=k, balance_labels=True)
            r = run_few_shot(llm, req, demos)
        elif method == "cot":
            demos = retriever.retrieve(req, k=k, balance_labels=True)
            r = run_cot(llm, req, demos, cot_cache)
        elif method == "marq":
            r = run_marq(llm, req, retriever, k=k)
        else:
            raise ValueError(f"unknown method {method}")
        latency_ms = int((time.time() - t0) * 1000)

        rec = {
            "fold": fold,
            "model": model_slug,
            "method": method,
            "requirement_id": int(row["requirement_id"]),
            "requirement_text": req,
            "weak_words_annotated": json.dumps(list(row["weak_words"])),
            "per_word_labels_annotated": json.dumps(dict(row["per_word_labels"])),
            "gold_label": row["label"],
            "pred_label": r["pred_label"],
            "reasoning": r["reasoning"],
            "n_llm_calls": r.get("n_llm_calls", 1),
            "input_tokens": r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
            "latency_ms": latency_ms,
            "retrieved_pool_ids": json.dumps(r.get("retrieved_pool_ids", [])),
        }
        records.append(rec)

        # incremental save every 50 to survive crashes
        if len(records) % 50 == 0:
            pd.DataFrame(records).to_csv(out_path, index=False)

    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"[saved] {out_path}  ({len(records)} predictions)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits/")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    ap.add_argument("--models", nargs="*", default=list(MODEL_REGISTRY.keys()),
                    choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--k", type=int, default=5, help="retrieval k for few-shot / cot / marq")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: fold 0, smallest model, 30 examples")
    args = ap.parse_args()

    if args.quick:
        args.folds = [0]
        args.models = ["qwen-1.5b"]
        args.methods = ["zero_shot", "marq"]

    splits_dir = Path(args.splits)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        pool, test, cot_cache = load_fold(splits_dir, fold)
        if args.quick:
            test = test.head(30).reset_index(drop=True)
        print(f"\n=== fold {fold}  pool={len(pool)}  test={len(test)} ===")
        retriever = PoolRetriever(pool, cache_key=f"fold{fold}")

        for model_key in args.models:
            model_name = MODEL_REGISTRY[model_key]
            for method in args.methods:
                if method == "cot" and not cot_cache:
                    print(f"[warn] no cot_chains.json for fold {fold}; "
                          f"run `python baselines.py prepare-cot` first. Skipping.")
                    continue
                run_one_method_one_fold(
                    method=method, model_name=model_name, model_slug=model_key,
                    fold=fold, test=test, retriever=retriever,
                    cot_cache=cot_cache, out_dir=out_dir, k=args.k,
                )

    print("\n[all done]")


if __name__ == "__main__":
    main()
