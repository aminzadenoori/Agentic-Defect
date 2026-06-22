"""
orchestrator.py — MARQ pipeline.

Pipeline:
  1. Retriever returns top-k pool examples.
  2. Scanner identifies candidate ambiguity tokens. If none → ok, stop.
  3. Investigator runs once per concern (sequential; could be parallelized).
  4. Critic reviews all investigator verdicts; may flag some for re-investigation.
  5. Investigator re-runs for challenged concerns (one round, no recursion).
  6. Synthesizer aggregates into a final requirement-level verdict + rationale.

Returns a dict with:
  pred_label, reasoning (joined trace), n_llm_calls, input_tokens,
  output_tokens, retrieved_pool_ids, trace (structured).
"""

from __future__ import annotations
from typing import Any, Dict, List

from agents import (
    scanner_agent, investigator_agent, critic_agent, synthesizer_agent,
)
from baselines import LLMBackend
from retrieval import PoolRetriever


def run_marq(llm: LLMBackend, requirement: str,
             retriever: PoolRetriever, k: int = 5,
             max_concerns: int = 5) -> Dict[str, Any]:
    trace: List[str] = []
    total_in = total_out = total_calls = 0

    # 1) retrieval
    demos = retriever.retrieve(requirement, k=k, balance_labels=True)
    trace.append(f"[RETRIEVER] k={k}, ids={[d['pool_id'] for d in demos]}")

    # 2) scanner
    concerns = scanner_agent(llm, requirement, demos)
    total_calls += 1
    trace.append(f"[SCANNER] found {len(concerns)} concern(s): "
                 + (", ".join(f"'{c['token']}'" for c in concerns) if concerns else "NONE"))

    if not concerns:
        # no concerns → ok, no further calls
        return {
            "pred_label": "ok",
            "reasoning": " || ".join(trace + ["[FAST-PATH] no concerns → ok"]),
            "n_llm_calls": total_calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "retrieved_pool_ids": [d["pool_id"] for d in demos],
            "trace": {"concerns": [], "investigators": [], "critic": None, "synth": None},
        }

    # cap concerns to keep cost predictable
    concerns = concerns[:max_concerns]

    # 3) investigator (sequential)
    invest_results: List[Dict[str, Any]] = []
    for c in concerns:
        ir = investigator_agent(llm, requirement, c["token"], c["reason"], demos)
        total_calls += 1
        total_in += ir["input_tokens"]; total_out += ir["output_tokens"]
        invest_results.append(ir)
        trace.append(f"[INVESTIGATOR] '{c['token']}' → {ir['verdict']} "
                     f"(conf {ir['confidence']}): {ir['rationale']}")

    # 4) critic
    critic_out = critic_agent(llm, requirement, invest_results)
    total_calls += 1
    total_in += critic_out["input_tokens"]; total_out += critic_out["output_tokens"]
    if critic_out["challenged"]:
        trace.append(f"[CRITIC] challenged indices {critic_out['challenged']}: "
                     + "; ".join(f"#{i}:{critic_out['notes'].get(i,'')}" for i in critic_out["challenged"]))
    else:
        trace.append("[CRITIC] no challenges")

    # 5) one re-investigation round if challenged
    if critic_out["challenged"]:
        for idx in critic_out["challenged"]:
            if 0 <= idx < len(invest_results):
                orig = invest_results[idx]
                extra_reason = (
                    f"{orig['rationale']}  [CRITIC NOTE: {critic_out['notes'].get(idx, '')}]"
                )
                ir2 = investigator_agent(llm, requirement, orig["token"], extra_reason, demos)
                total_calls += 1
                total_in += ir2["input_tokens"]; total_out += ir2["output_tokens"]
                invest_results[idx] = ir2
                trace.append(f"[INVESTIGATOR-rev] '{orig['token']}' → {ir2['verdict']} "
                             f"(conf {ir2['confidence']}): {ir2['rationale']}")

    # 6) synthesizer
    synth = synthesizer_agent(llm, requirement, invest_results)
    total_calls += 1
    total_in += synth["input_tokens"]; total_out += synth["output_tokens"]
    trace.append(f"[SYNTH] verdict={synth['verdict']} :: {synth['summary']}")

    return {
        "pred_label": synth["verdict"],
        "reasoning": " || ".join(trace),
        "n_llm_calls": total_calls,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "retrieved_pool_ids": [d["pool_id"] for d in demos],
        "trace": {
            "concerns": concerns,
            "investigators": invest_results,
            "critic": critic_out,
            "synth": synth,
        },
    }
