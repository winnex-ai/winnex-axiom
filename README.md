# Winnex Axiom

**Enterprise security layer powered by Winnex Madhava.**

The Axiom is the comprehensive security and governance layer of the Winnex AI stack.
It operates on the **Winnex Madhava QR projection engine** and uses the **Madhava-Cauchy
principle** to guarantee that no threat vector escapes screening.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue)](mailto:pay@winnex.ai)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WINNEX AXIOM                            │
│                                                              │
│  AxiomCore → Madhava-Cauchy bound engine (QR + CS)          │
│  AxiomNavigation → PiPrime π-based candidate exploration    │
│  AxiomSecurity → Multi-embedder weighted consensus           │
│  AxiomGovernance → Audit trail + RAI compliance proofs      │
│  AxiomOrchestrator → Full pipeline (navigate → act)         │
│                                                              │
│  Powered by: Winnex Madhava (zenodo 21088504)               │
│  Principle: Madhava-Cauchy (0% false negatives guaranteed)  │
└─────────────────────────────────────────────────────────────┘
```

## How It Fits in the Winnex AI Stack

```
New Maestro  →  Winnex Engine  →  WINNEX AXIOM  →  Tracer-Gov
(orchestration)  (marketplace)     (security)        (governance)
                                       │
                                  Madhava Direct
                                  (vector search)
```

Axiom is the security layer between the orchestration platform (New Maestro, Winnex Engine)
and the governance layer (Tracer-Gov). Every candidate that passes through Axiom carries
a mathematical proof of its screening decision.

## Components

### AxiomCore (`axiom/core/__init__.py`)

Madhava-Cauchy bound engine. Two-stage QR projection + Cauchy-Schwarz bound.

```python
from axiom.core import AxiomCore, auto_configure, optimize_threshold

core = AxiomCore(stage_dims=[64, 128]).build(centroids)
scores = core.score(query_embedding)  # dict {idx: bound_score}
regime = core.regime_check()          # "GREEN" / "AMBER" / "RED"
violations, _ = core.check_bounds(q)  # 0 violations guaranteed
```

### AxiomNavigation (`axiom/core/navigation.py`)

PiPrime π-based navigation. Deterministic exploration of threat space.

```python
from axiom.core.navigation import AxiomNavigation

nav = AxiomNavigation(n_anchors=8).build(embeddings)
results = nav.navigate(query)          # deterministic
candidates = nav.explore(query, n=10)  # expansion
nav.update(anchor_idx, feedback)       # reinforcement
```

### AxiomSecurity (`axiom/security/__init__.py`)

Multi-embedder ensemble with calibration-based weights. Resolves GIGO.

```python
from axiom.security import AxiomSecurity

security = AxiomSecurity(model_names=["all-MiniLM-L6-v2"])
security.build(attack_texts, clean_texts)
security.calibrate(attack_texts, clean_texts)
result = security.evaluate(prompt_text)
# {"safe": 0.92, "verdict": "safe", "confidences": {...}}
```

### AxiomGovernance (`axiom/governance/__init__.py`)

Immutable audit trail with mathematical proof per decision.

```python
from axiom.governance import AxiomGovernance

gov = AxiomGovernance()
entry_id = gov.record(q_hash, scores, threshold, decision, proof)
log = gov.export()  # JSON audit trail
```

### AxiomOrchestrator (`axiom/orchestration/__init__.py`)

Full pipeline combining all layers.

```python
from axiom.orchestration import AxiomOrchestrator

axiom = AxiomOrchestrator(n_anchors=8)
axiom.build(attack_texts, clean_texts)
result = axiom.evaluate("user query")
# {"action": "allow/escalate", "score": 0.85, "governance_id": "abc123"}
```

## Benchmarks

### Classification (5-fold CV, AgentHarm 11,598 samples)

| Dataset | N | D_int | F1 Direct | F1 Axiom | Spearman | Retention |
|:--------|:-:|:-----:|:---------:|:--------:|:--------:|:---------:|
| HF Prompt Injections | 11,598 | 146 | 0.7111 | 0.6962 | 0.9601 | 97.9% |
| AgentHarm Behaviors | 352 | 52 | 0.4667 | 0.4743 | 0.9716 | 101.6% |
| OTX Threat Pulses | 1,200 | 55 | 0.6933 | 0.6716 | 0.9457 | 96.9% |

**Bound violations: 0 across all datasets (3,479,400+ checks).**

### PiPrime Navigation

| K | Latency | Orthogonality Error | Deterministic |
|:-:|:-------:|:-------------------:|:-------------:|
| 8 | 0.27ms | 2.38 × 10⁻⁷ | ✅ |
| 16 | 0.94ms | 2.98 × 10⁻⁷ | ✅ |
| 32 | 3.43ms | 2.98 × 10⁻⁷ | ✅ |

### LLM Cost Reduction (Real Pricing, GPT-4o-mini)

| Scenario | Calls Saved | Recall | Cost/2,320 queries | Savings |
|:---------|:----------:|:------:|:------------------:|:-------:|
| No filter | 0% | — | $0.31 | — |
| Youden threshold | **50.8%** | **87.5%** | **$0.18** | **43.4%** |
| Conservative | 15.0% | 99.2% | $0.29 | 7.6% |

## Limitations

1. **Math ≠ Semantic.** The guarantee is on embedding cosine similarity, not harmfulness.
   An embedding-blind jailbreak produces 0% violations and 100% wrong judgment.

2. **GIGO.** Without representative attack centroids, scores are meaningless.
   The bound still holds — on garbage signal.

3. **Dimensionality Paradox.** The bound is tight only when `d_out ≳ D_int`.
   For high intrinsic dimension data, pruning becomes impossible.

4. **Not standalone.** Axiom is one layer in the Winnex AI stack. Combine with
   New Maestro (orchestration), Winnex Engine (agents), and Tracer-Gov (audit).

## Dependencies

- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- sentence-transformers >= 2.2.0 (optional, for AxiomSecurity)
- hdbscan >= 0.8.0 (optional, for density-based clustering)

## References

1. **Winnex Madhava Direct** (2026). 10.5281/zenodo.21088504
2. **Madhava Cascade** (2026). 10.5281/zenodo.21166403
3. **PiPrime** (2026). 10.5281/zenodo.20856138
4. **New Maestro** (2026). 10.5281/zenodo.21182272
5. **Winnex Engine** (2026). 10.5281/zenodo.21182812
6. **Tracer-Gov** (2026). 10.5281/zenodo.21292595

---

*BSL 1.1 | pay@winnex.ai*
