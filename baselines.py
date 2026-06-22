"""
baselines.py — Zero-shot, Few-shot (k=5), CoT few-shot (k=5).

Every method returns a (pred_label, reasoning, n_llm_calls) triple. The
reasoning string is what RQ5's human study will read.

Shared backbone: LLMBackend (wraps an HF AutoModelForCausalLM with chat
template + greedy decoding).
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


VERDICT_RE = re.compile(r"verdict\s*:\s*(ok|defect)", re.IGNORECASE)
FALLBACK_RE = re.compile(r"\b(defect|ok)\b", re.IGNORECASE)


def parse_verdict(text: str) -> str:
    m = VERDICT_RE.search(text)
    if m:
        return m.group(1).lower()
    m = FALLBACK_RE.search(text)
    return m.group(1).lower() if m else "ok"


# =================================================================
# LLM backend
# =================================================================

@dataclass
class GenResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMBackend:
    """Shared, lazy-loaded backbone used by all baselines and all agents."""

    _cache: Dict[str, "LLMBackend"] = {}

    @classmethod
    def get(cls, model_name: str, device: str = "cuda") -> "LLMBackend":
        if model_name not in cls._cache:
            cls._cache[model_name] = cls(model_name, device)
        return cls._cache[model_name]

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=self.device,
        )
        self.model.eval()

    @torch.no_grad()
    def chat(self, messages: List[Dict[str, str]],
             max_new_tokens: int = 256, temperature: float = 0.0) -> GenResult:
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=4096).to(self.device)
        n_in = enc["input_ids"].shape[1]
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=None,
            pad_token_id=self.tok.pad_token_id,
        )
        gen = out[:, n_in:]
        n_out = gen.shape[1]
        text = self.tok.decode(gen[0], skip_special_tokens=True).strip()
        return GenResult(text=text, input_tokens=int(n_in), output_tokens=int(n_out))


# =================================================================
# Prompt builders
# =================================================================

SYSTEM = (
    "You are an experienced industrial software requirements reviewer. "
    "Your job is to decide whether a requirement contains an ambiguity defect "
    "(vague or unmeasurable language with no grounding in the context) or is "
    "acceptable as written. Be concise but explain your reasoning."
)

INSTRUCTION_ZERO = (
    "Read the requirement below and decide whether it contains an ambiguity defect.\n\n"
    "A requirement is a DEFECT if it contains vague terms (e.g., 'adequate', "
    "'appropriate', 'short', 'fast', 'soon', 'several') that are NOT grounded by "
    "specific values, named references, illustrative examples, or measurable criteria.\n\n"
    "A requirement is OK if any potentially vague terms are clearly grounded by context "
    "OR if the requirement contains no vague terms at all.\n\n"
    "Reply in exactly this format:\n"
    "Reasoning: <one or two sentences>\n"
    "Verdict: ok | defect"
)


def _format_demo(ex: Dict[str, Any], include_cot: bool = False, cot_text: str | None = None) -> str:
    """Single in-context demo."""
    annot = ", ".join(f"'{w}': {ex['per_word_labels'][w]}" for w in ex["weak_words"])
    s = (
        f'Requirement: "{ex["requirement"]}"\n'
        f"(Annotated weak words: {annot})\n"
    )
    if include_cot and cot_text:
        s += f"Reasoning: {cot_text}\n"
    else:
        s += "Reasoning: " + _build_reason_from_annotations(ex) + "\n"
    s += f"Verdict: {ex['label']}"
    return s


def _build_reason_from_annotations(ex: Dict[str, Any]) -> str:
    """Heuristic reasoning when no synthesized CoT chain is available."""
    if ex["label"] == "defect":
        defect_words = [w for w, l in ex["per_word_labels"].items() if l == "defect"]
        return (f"The word(s) {defect_words!r} appear without specific grounding "
                f"(no named values, references, or measurable criteria).")
    else:
        return ("All weak words present are grounded by context — named values, "
                "references, illustrative examples, or precise technical use.")


def build_messages_zero(requirement: str) -> List[Dict[str, str]]:
    user = (
        f"{INSTRUCTION_ZERO}\n\n"
        f'Requirement: "{requirement}"\n\n'
        "Reasoning:"
    )
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


def build_messages_fewshot(requirement: str, demos: List[Dict[str, Any]],
                            include_cot: bool = False,
                            cot_cache: Dict[int, str] | None = None) -> List[Dict[str, str]]:
    demo_block = "\n\n".join(
        _format_demo(d, include_cot=include_cot,
                     cot_text=(cot_cache or {}).get(d["pool_id"]))
        for d in demos
    )
    user = (
        f"{INSTRUCTION_ZERO}\n\n"
        "Here are labeled examples from prior reviews:\n\n"
        f"{demo_block}\n\n"
        "Now classify this new requirement using the same reasoning style.\n\n"
        f'Requirement: "{requirement}"\n\n'
        "Reasoning:"
    )
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


# =================================================================
# Method runners
# =================================================================

def parse_reasoning_and_verdict(text: str) -> Tuple[str, str]:
    """Extract (reasoning, verdict) from a model output."""
    # split on "Verdict:"
    m = re.search(r"(?is)(.*?)\bverdict\s*:\s*(ok|defect)", text)
    if m:
        reasoning = m.group(1).strip().lstrip(":").strip()
        # strip leading "Reasoning:"
        reasoning = re.sub(r"^reasoning\s*:\s*", "", reasoning, flags=re.IGNORECASE).strip()
        verdict = m.group(2).lower()
        return reasoning, verdict
    return text.strip(), parse_verdict(text)


def run_zero_shot(llm: LLMBackend, requirement: str) -> Dict[str, Any]:
    msgs = build_messages_zero(requirement)
    r = llm.chat(msgs, max_new_tokens=180)
    reasoning, verdict = parse_reasoning_and_verdict(r.text)
    return {
        "pred_label": verdict,
        "reasoning": reasoning,
        "n_llm_calls": 1,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "raw_output": r.text,
    }


def run_few_shot(llm: LLMBackend, requirement: str, demos: List[Dict[str, Any]]) -> Dict[str, Any]:
    msgs = build_messages_fewshot(requirement, demos, include_cot=False)
    r = llm.chat(msgs, max_new_tokens=220)
    reasoning, verdict = parse_reasoning_and_verdict(r.text)
    return {
        "pred_label": verdict,
        "reasoning": reasoning,
        "n_llm_calls": 1,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "raw_output": r.text,
        "retrieved_pool_ids": [d["pool_id"] for d in demos],
    }


def run_cot(llm: LLMBackend, requirement: str, demos: List[Dict[str, Any]],
            cot_cache: Dict[int, str]) -> Dict[str, Any]:
    msgs = build_messages_fewshot(requirement, demos, include_cot=True, cot_cache=cot_cache)
    r = llm.chat(msgs, max_new_tokens=320)
    reasoning, verdict = parse_reasoning_and_verdict(r.text)
    return {
        "pred_label": verdict,
        "reasoning": reasoning,
        "n_llm_calls": 1,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "raw_output": r.text,
        "retrieved_pool_ids": [d["pool_id"] for d in demos],
    }


# =================================================================
# CoT chain synthesis (one-time, cached)
# =================================================================

def synthesize_cot_chains(pool_df, teacher_model: str, out_path: str) -> Dict[int, str]:
    """Generate one reasoning chain per pool example. Run once per pool fold."""
    import json
    from pathlib import Path
    out = Path(out_path)
    if out.exists():
        return json.loads(out.read_text())

    llm = LLMBackend.get(teacher_model)
    cache: Dict[int, str] = {}
    for _, row in pool_df.iterrows():
        annot = ", ".join(f"'{w}': {row['per_word_labels'][w]}" for w in row["weak_words"])
        user = (
            f'Requirement: "{row["requirement"]}"\n'
            f"Annotated weak words and their per-word labels: {annot}\n"
            f"Aggregate label: {row['label']}\n\n"
            "In ONE sentence, explain why this aggregate label is correct. Focus on whether "
            "the weak word(s) are grounded by specific context. Do not restate the requirement."
        )
        r = llm.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            max_new_tokens=80,
        )
        cache[int(row["requirement_id"])] = r.text.strip().split("\n")[0]
    out.write_text(json.dumps(cache, indent=2))
    return cache


# =================================================================
# CLI for one-time CoT prep
# =================================================================

if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare-cot"])
    p.add_argument("--pool", required=True, help="splits/ directory")
    p.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    args = p.parse_args()

    splits_dir = Path(args.pool)
    for fold_dir in sorted(splits_dir.glob("fold_*")):
        import pandas as pd
        pool_df = pd.read_parquet(fold_dir / "pool.parquet")
        out_path = fold_dir / "cot_chains.json"
        print(f"[cot] {fold_dir} → {out_path} (n={len(pool_df)})")
        synthesize_cot_chains(pool_df, args.teacher, str(out_path))
