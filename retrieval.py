"""
retrieval.py — Dense top-k retriever over a fold's pool.

All four methods (zero-shot, few-shot, CoT, MARQ) use this same retriever for
fair comparison. Zero-shot ignores the retrieved examples; the others use them.

Embeddings are cached per (pool_path, model_name) on disk to avoid recomputation.

Usage:
    from retrieval import PoolRetriever
    r = PoolRetriever(pool_df)
    examples = r.retrieve("The system shall respond fast.", k=5)
"""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(".cache/embeddings")


def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


class PoolRetriever:
    def __init__(self, pool_df: pd.DataFrame,
                 embed_model: str = DEFAULT_EMBED_MODEL,
                 cache_key: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.pool_df = pool_df.reset_index(drop=True)
        self.embed_model_name = embed_model
        self.encoder = SentenceTransformer(embed_model)

        # cache embeddings on disk
        if cache_key is None:
            cache_key = _md5(str(pool_df["requirement"].tolist()) + embed_model)
        cache_path = CACHE_DIR / f"pool_{cache_key}.npy"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            self.pool_emb = np.load(cache_path)
        else:
            self.pool_emb = self.encoder.encode(
                pool_df["requirement"].tolist(),
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False, batch_size=64,
            )
            np.save(cache_path, self.pool_emb)

    def retrieve(self, query: str, k: int = 5,
                 balance_labels: bool = True) -> List[Dict[str, Any]]:
        """Return top-k pool examples by cosine similarity.

        If balance_labels=True, attempt to return roughly half ok, half defect
        (improves few-shot calibration). Falls back to plain top-k if pool is
        too imbalanced.
        """
        q_emb = self.encoder.encode([query], normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        sims = (self.pool_emb @ q_emb.T).ravel()

        if balance_labels:
            k_each = k // 2
            picks = []
            for lab in ("ok", "defect"):
                mask = (self.pool_df["label"] == lab).values
                if mask.sum() == 0:
                    continue
                lab_idx = np.where(mask)[0]
                lab_sims = sims[lab_idx]
                top = lab_idx[np.argsort(-lab_sims)[:k_each]]
                picks.extend(top.tolist())
            # fill remainder by top-k of whatever's left
            remaining = k - len(picks)
            if remaining > 0:
                all_top = np.argsort(-sims)
                for i in all_top:
                    if i not in picks:
                        picks.append(int(i))
                        if len(picks) >= k:
                            break
        else:
            picks = np.argsort(-sims)[:k].tolist()

        examples = []
        for i in picks[:k]:
            row = self.pool_df.iloc[i]
            examples.append({
                "pool_id": int(row["requirement_id"]),
                "requirement": row["requirement"],
                "weak_words": list(row["weak_words"]),
                "per_word_labels": dict(row["per_word_labels"]),
                "label": row["label"],
                "similarity": float(sims[i]),
            })
        return examples
