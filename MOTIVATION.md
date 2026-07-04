# Motivation and Hypothesis

*Companion note to the MARQ paper. This document develops the scientific
argument for why the task calls for a multi-agent architecture rather than a
better single prompt. It is the source material for the paper's "Motivation"
section; the README states the research questions that operationalize it.*

---

## 1. The objection we must answer

A reviewer will accept RQ1 as an empirical sanity check — if the proposed
architecture were not at least competitive with prompt-based baselines, there
would be no reason to study it. But competitiveness is not a *motivation*. The
harder question is: **why should a multi-agent system be needed at all, if
carefully engineered prompting already performs strongly?** If zero-shot,
few-shot, and chain-of-thought (CoT) prompting are close, a substantially more
complex architecture is not obviously justified. A referee can reasonably ask
why a stronger CoT prompt, or CoT with self-consistency, would not close the
gap just as well.

Our answer has two parts. First, the setting we study is genuinely harder than
the one under which prompt baselines are usually evaluated. Second — and this is
the part that specifically motivates *agents* rather than merely *more
reasoning* — the task has structural properties that monolithic prompting cannot
satisfy in principle, regardless of how the prompt is engineered.

---

## 2. The setting is harder than published QuRE

QuRE provides labeled `(requirement, weak_word) -> ok/defect` pairs: the model
is *told* which word to inspect. Real reviewers are not. We reformulate to the
requirement level — a requirement is `defect` if *any* of its weak-word
instances is `defect`, and `ok` only if all are — and we **hide the weak-word
annotation at test time**. The model must first decide *where* to look before it
can decide *whether* what it finds is problematic.

This reformulation is necessary but not sufficient as a motivation. It justifies
*decomposing* the task into "locate, then judge"; it does not yet explain why
that decomposition should be realized as *separate agents* rather than as two
phases of a single prompt. The next section supplies that missing step.

---

## 3. Why structure, not just more reasoning — three properties of the task

We identify three properties of requirement-level defect detection that a
single reasoning trajectory cannot satisfy, and that a structured multi-agent
decomposition addresses directly. These are properties of the *problem*, not
of any particular model, which is what makes them a defensible motivation.

### 3.1 Conflicting sub-objectives

Once the weak word is hidden, the task splits into two subtasks with *opposite*
optimal operating points. Locating candidate ambiguous expressions is a
**recall** problem: missing a problematic span guarantees a wrong label, so the
locating step should over-generate candidates. Judging whether a candidate is
actually ambiguous in context is a **precision** problem: flagging every hedge
word as a defect would destroy accuracy, so the judging step should be
conservative. A monolithic prompt exposes a single instruction set and a single
effective operating point; it cannot be simultaneously recall-oriented in
detection and precision-oriented in judgment. Decomposition lets the locating
agent run "hot" (favor recall) and the judging agent run "cold" (favor
precision). No amount of prompt wording removes this tension inside one call.

### 3.2 Decorrelated verification (the property that separates us from self-consistency)

CoT keeps the model on a single reasoning trajectory. Self-consistency samples
*many* trajectories from the *same* distribution and votes — which reduces
variance but not systematic bias: if the model is confidently wrong, its samples
tend to be wrong in correlated ways, and the majority vote inherits the error.
Catching a confidently-wrong verdict requires an assessment whose errors are
*decorrelated* from the generator's. A separate critic agent, prompted with a
different role and *without* having produced the original verdict, is not
anchored to the generation trajectory and does not rationalize it. This is why
the paper's hypothesis is not "more reasoning helps" but "**decorrelated,
independently-verified reasoning helps**" — a claim self-consistency does not
satisfy.

### 3.3 Context isolation and auditability

Handling several candidate concerns inside one prompt forces the model to reason
about all of them in a shared context, where the analysis of one span can leak
into and bias the analysis of another. Assigning one investigator per concern
isolates each judgment and gives it full attention. It also yields a *per-concern*
rationale rather than a single entangled reasoning blob — which is precisely
what the RQ5 benchmark-audit study needs a human adjudicator to read. The
architecture is therefore co-designed with the auditing goal, not only with
accuracy: a property a monolithic prompt cannot offer.

---

## 4. A running example

Consider the requirement:

> *"The system shall process the incoming request quickly and return an
> appropriate response to the user."*

Two expressions are potentially problematic: **"quickly"** (unquantified
performance) and **"appropriate"** (subjective, context-dependent). Under
published QuRE the model would be handed one of these words and asked only to
rate it. Under our setting it receives the sentence alone.

A human reviewer does not decide "ambiguous / not ambiguous" in one step. They:

1. **Scan** the sentence for expressions that *could* be underspecified —
   surfacing "quickly" and "appropriate", and perhaps considering and
   discarding "the user".
2. **Assess each candidate in context, separately.** "Quickly" has no bound
   and no reference workload — likely a genuine defect. "Appropriate response"
   might be acceptable if an earlier requirement defines the response set — so
   the verdict is conditional and needs the surrounding evidence.
3. **Challenge the initial reading** when the evidence is inconclusive. Is
   "quickly" perhaps constrained by a non-functional requirement elsewhere? Is
   "appropriate" a defined term? A good reviewer actively tries to falsify their
   own first impression before committing.
4. **Integrate** the surviving concerns into one verdict: any unresolved
   ambiguity makes the requirement a defect.

MARQ mirrors this process **not because imitating humans is inherently correct,
but because each of these steps is where one of the three task properties above
bites**: step 1 is the recall-oriented locating subtask (3.1); steps 2 handled
per concern is context isolation (3.3); step 3 is decorrelated verification
(3.2); step 4 is the deterministic aggregation rule. The human walkthrough is an
intuition pump; the justification for each agent is the computational property
it secures.

---

## 5. Why these specific agents, and this division of labor

| Agent | Functional justification (not "it mimics a human") |
|---|---|
| **Retriever** | Deterministic dense top-k over the labeled pool. Supplies what the hidden annotation would otherwise provide: examples of what "weak" looks like, learned from data rather than a fixed lexicon. |
| **Scanner** | Owns the **recall-oriented locating** subtask (3.1). Its job is to not miss candidate spans; precision is deferred downstream. |
| **Investigator** (one per concern, parallel) | Owns the **precision-oriented judging** subtask (3.1) under **context isolation** (3.3). One concern per call prevents cross-contamination and yields per-concern rationales for RQ5. |
| **Critic** | The **decorrelated verifier** (3.2). Challenges low-confidence verdicts and verdicts that contradict retrieved demonstrations; may trigger one re-investigation round. This is the agent that most distinguishes MARQ from a prompt pipeline, so its rationale must be stated most explicitly. |
| **Synthesizer** | Applies the transparent aggregation rule (any defect -> defect). Deliberately *not* where reasoning happens, so the decision rule stays inspectable. |

The division of labor is therefore driven by (a) the two-stage structure of the
reformulated task, (b) the conflicting recall/precision objectives, and (c) the
need for verification whose errors do not correlate with the generator's — with
the human-review analogy used only to make the structure intuitive.

---

## 6. Relationship to the baselines

The baselines are not strawmen; they are the successively stronger monolithic
alternatives the hypothesis must beat.

- **Zero-shot** and **few-shot** perform the whole task in one instruction set —
  they cannot separate recall from precision (3.1) and offer no verification
  (3.2). Evidence from prompt-engineering studies (e.g., "prompt smells")
  further suggests that packing many instructions into one prompt tends to
  *degrade* performance, which is a symptom of 3.1, not its cause.
- **CoT** adds intermediate steps but keeps them on a **single trajectory
  produced by a single agent**: intermediate hypotheses are neither
  independently verified nor explicitly challenged before the final decision.
  CoT improves reasoning depth but does not provide 3.2.
- **CoT + self-consistency** (recommended addition; see below) is the sharpest
  cheap alternative, and the one a reviewer will name. It reduces variance by
  voting over trajectories from the same distribution, but does **not**
  decorrelate errors from the generator — so by construction it does not satisfy
  3.2. Including it turns "MARQ beats monolithic reasoning" from a rhetorical
  claim into a measured one.

**Recommendation.** Add CoT + self-consistency as a fifth method. Without it,
the objection "you never tried the obvious cheaper thing" stands regardless of
how strong the motivation section reads. If added, the "Methods compared"
section and the run-count arithmetic in the README (currently 4 x 5 x 5 = 100)
must be updated accordingly.

---

## 7. The hypothesis, stated for the paper

> **H1.** For requirement-level ambiguity-defect detection *without* a supplied
> weak-word hint, structured multi-agent reasoning — which separates
> recall-oriented location from precision-oriented judgment and subjects
> intermediate verdicts to decorrelated, independent verification before
> aggregation — yields higher detection quality than monolithic
> single-trajectory prompting (zero-shot, few-shot, CoT, and CoT with
> self-consistency), at a cost that remains within a practically useful region
> of the cost-quality frontier.

The claim is deliberately *not* "more reasoning is better." It is that
**structured, decorrelated reasoning is better matched to this class of
requirements-analysis tasks than monolithic reasoning carried out within a
single prompt** — and RQ1–RQ4 are designed to test exactly that, while RQ5 turns
the resulting reasoning traces into an audit of the benchmark itself.
