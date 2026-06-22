"""
agents.py — Four specialized agents that share a single LLM backbone.

Each agent has its own system prompt and a tight output contract that the
orchestrator parses. All agents see the same retrieved examples; what differs
is what they're asked to do with them.

Output contracts (parsed by orchestrator):
  Scanner    : numbered list of "TOKEN: <word> | REASON: <one line>"
  Investigator: "VERDICT: ok|defect" + "CONFIDENCE: 1-5" + "RATIONALE: ..."
  Critic     : list of indices to reconsider + optional new evidence
  Synthesizer: "VERDICT: ok|defect" + "SUMMARY: ..."
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from baselines import LLMBackend


# =================================================================
# Shared formatting helpers
# =================================================================

def _format_pool_demo(ex: Dict[str, Any]) -> str:
    annot = ", ".join(f"'{w}': {ex['per_word_labels'][w]}" for w in ex["weak_words"])
    return f"- \"{ex['requirement']}\"  | weak words: {annot}  | label: {ex['label']}"


def _examples_block(demos: List[Dict[str, Any]]) -> str:
    return "Relevant labeled examples from prior reviews:\n" + "\n".join(
        _format_pool_demo(d) for d in demos
    )


# =================================================================
# Scanner agent
# =================================================================

SCANNER_SYSTEM = (
    "You are a requirements quality SCANNER. Read a software requirement and identify "
    "any tokens or short phrases that might be ambiguity defects — vague terms that lack "
    "measurable criteria. Examples of typical weak tokens include: 'adequate', "
    "'appropriate', 'short', 'fast', 'soon', 'several', 'reasonable', 'sufficient'.\n\n"
    "You will be shown labeled examples of similar requirements. Use them to calibrate.\n\n"
    "Output a numbered list. Each item must be exactly:\n"
    "  N. TOKEN: <word or short phrase> | REASON: <one short sentence>\n\n"
    "If you find no candidates, output exactly: NONE"
)


def scanner_agent(llm: LLMBackend, requirement: str, demos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    user = (
        f"{_examples_block(demos)}\n\n"
        f"Now scan this requirement:\n"
        f'"{requirement}"\n\n'
        "List the candidate ambiguity tokens (or NONE):"
    )
    r = llm.chat(
        [{"role": "system", "content": SCANNER_SYSTEM},
         {"role": "user", "content": user}],
        max_new_tokens=220,
    )
    text = r.text.strip()
    concerns = []
    if text.upper().startswith("NONE"):
        return concerns
    for line in text.splitlines():
        m = re.match(r"\s*\d+\.\s*TOKEN\s*:\s*(.+?)\s*\|\s*REASON\s*:\s*(.+)$",
                     line, flags=re.IGNORECASE)
        if m:
            concerns.append({
                "token": m.group(1).strip().strip("'\""),
                "reason": m.group(2).strip(),
            })
    return concerns


# =================================================================
# Investigator agent (one call per concern)
# =================================================================

INVESTIGATOR_SYSTEM = (
    "You are a requirements quality INVESTIGATOR. You will receive a requirement and "
    "ONE specific token of concern. Decide whether that token is grounded in the "
    "requirement's context.\n\n"
    "A token is GROUNDED (verdict 'ok') if any of these is true:\n"
    "  - a specific numeric value or unit is given;\n"
    "  - a named standard or reference document is cited;\n"
    "  - an illustrative example (e.g., 'such as X', 'e.g., Y') is provided;\n"
    "  - the token is used in a precise technical sense (e.g., 'short circuit').\n\n"
    "Otherwise the token is NOT grounded (verdict 'defect').\n\n"
    "Output exactly three lines:\n"
    "  VERDICT: ok | defect\n"
    "  CONFIDENCE: 1-5\n"
    "  RATIONALE: <one short sentence>"
)


def investigator_agent(llm: LLMBackend, requirement: str, token: str,
                       reason: str, demos: List[Dict[str, Any]]) -> Dict[str, Any]:
    user = (
        f"{_examples_block(demos)}\n\n"
        f"Requirement under review:\n\"{requirement}\"\n\n"
        f"Token of concern: '{token}'\n"
        f"Why it was flagged: {reason}\n\n"
        "Investigate:"
    )
    r = llm.chat(
        [{"role": "system", "content": INVESTIGATOR_SYSTEM},
         {"role": "user", "content": user}],
        max_new_tokens=120,
    )
    text = r.text
    verdict = "ok"
    confidence = 3
    rationale = text.strip()
    m_v = re.search(r"verdict\s*:\s*(ok|defect)", text, re.IGNORECASE)
    if m_v: verdict = m_v.group(1).lower()
    m_c = re.search(r"confidence\s*:\s*([1-5])", text, re.IGNORECASE)
    if m_c: confidence = int(m_c.group(1))
    m_r = re.search(r"rationale\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m_r: rationale = m_r.group(1).strip().split("\n")[0]
    return {
        "token": token,
        "verdict": verdict,
        "confidence": confidence,
        "rationale": rationale,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
    }


# =================================================================
# Critic agent
# =================================================================

CRITIC_SYSTEM = (
    "You are an adversarial CRITIC for a requirements quality system. You receive a "
    "list of per-token verdicts. Your job is to challenge any verdict that looks weak.\n\n"
    "Challenge a verdict if:\n"
    "  - it was made with low confidence (1-2);\n"
    "  - the rationale ignores a specific value or reference that IS in the requirement;\n"
    "  - the rationale invents a reference that is NOT in the requirement.\n\n"
    "Output exactly one line:\n"
    "  CHALLENGE: <comma-separated indices, or NONE>\n"
    "Then for each challenged index, on its own line:\n"
    "  <index>: <one short sentence stating the new evidence or counter-argument>"
)


def critic_agent(llm: LLMBackend, requirement: str,
                 investigator_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not investigator_results:
        return {"challenged": [], "notes": {}, "input_tokens": 0, "output_tokens": 0}
    listing = "\n".join(
        f"{i}. TOKEN: '{r['token']}' | VERDICT: {r['verdict']} | "
        f"CONFIDENCE: {r['confidence']} | RATIONALE: {r['rationale']}"
        for i, r in enumerate(investigator_results)
    )
    user = (
        f'Requirement: "{requirement}"\n\n'
        f"Investigator verdicts:\n{listing}\n\n"
        "Critique:"
    )
    r = llm.chat(
        [{"role": "system", "content": CRITIC_SYSTEM},
         {"role": "user", "content": user}],
        max_new_tokens=180,
    )
    text = r.text
    challenged, notes = [], {}
    m = re.search(r"CHALLENGE\s*:\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        s = m.group(1).strip()
        if s.upper() != "NONE":
            for tok in re.findall(r"\d+", s):
                challenged.append(int(tok))
    for line in text.splitlines():
        nm = re.match(r"^\s*(\d+)\s*:\s*(.+)$", line)
        if nm and int(nm.group(1)) in challenged:
            notes[int(nm.group(1))] = nm.group(2).strip()
    return {
        "challenged": challenged, "notes": notes,
        "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
    }


# =================================================================
# Synthesizer agent
# =================================================================

SYNTHESIZER_SYSTEM = (
    "You are the SYNTHESIZER for a multi-agent requirements quality system. You receive "
    "the final per-token verdicts (after any critic-driven revisions) and must produce a "
    "single aggregate verdict for the requirement.\n\n"
    "Logic:\n"
    "  - If ANY per-token verdict is 'defect', the aggregate is 'defect'.\n"
    "  - If ALL per-token verdicts are 'ok' (or there were no concerns), the aggregate is 'ok'.\n\n"
    "Output exactly two lines:\n"
    "  VERDICT: ok | defect\n"
    "  SUMMARY: <one or two sentences summarizing why>"
)


def synthesizer_agent(llm: LLMBackend, requirement: str,
                      investigator_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not investigator_results:
        listing = "No concerns were raised by the scanner."
    else:
        listing = "\n".join(
            f"- TOKEN: '{r['token']}' → {r['verdict']} ({r['rationale']})"
            for r in investigator_results
        )
    user = (
        f'Requirement: "{requirement}"\n\n'
        f"Final per-token verdicts:\n{listing}\n\n"
        "Synthesize:"
    )
    r = llm.chat(
        [{"role": "system", "content": SYNTHESIZER_SYSTEM},
         {"role": "user", "content": user}],
        max_new_tokens=120,
    )
    text = r.text
    verdict = "ok"
    summary = text.strip()
    m_v = re.search(r"verdict\s*:\s*(ok|defect)", text, re.IGNORECASE)
    if m_v: verdict = m_v.group(1).lower()
    m_s = re.search(r"summary\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m_s: summary = m_s.group(1).strip()
    # safety-net: if any investigator said defect, force defect
    if any(ir["verdict"] == "defect" for ir in investigator_results):
        verdict = "defect"
    return {
        "verdict": verdict, "summary": summary,
        "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
    }
