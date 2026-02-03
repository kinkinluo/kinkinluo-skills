---
name: model-reproduction
description: |
  Strict skill for reproducing ML models. Use for: (1) Reproducing paper algorithms, (2) Porting open-source models, (3) Integrating multiple algorithms into one agent, (4) Adapting a model to downstream tasks. Core rule: engineering code may change (adapters, I/O, logging, config), but the algorithm must stay untouched.
  Keywords: reproduce, reimplement, port, migrate, adapt, integration.
---

# Model Reproduction

## Core Principle: Engineering Can Change, Algorithm Cannot

Always separate two code types:

| Type | Allowed to change | Typical content |
|------|-------------------|------------------|
| Engineering code | Yes | Data loading, I/O adapters, framework bridges, logging, config management |
| Algorithm code | No | Network architecture, loss formulation, optimization logic, math ops, default hyperparameters |

## Do-Not List

Never do any of the following:

1. Simplify the network: remove layers, reduce channels, delete residuals.
2. Change math formulas: alter loss, attention, normalization, or other core computations.
3. Swap core components: replace custom ops with generic ones; use approximations instead of exact logic.
4. Change default hyperparameters: lr, batch size, epochs/steps, weight decay, clipping thresholds.
5. Omit implementation details: skip init schemes, ignore gradient clipping, drop required data augmentation.
6. Use placeholders: no `pass`, stubs, or TODOs inside algorithm code.

## Reproduction Workflow

### Phase 1: Source Analysis (mandatory first)

```
1. Read the original code end-to-end; inventory all files.
2. Tag files/blocks as [Algorithm Core] vs [Engineering].
3. List every hyperparameter with its default value.
4. Record special details (initialization, regularization, precision, scheduling).
```

### Phase 2: Plan Before Coding

Report to the user, then start coding:

```markdown
## Reproduction Plan

### Algorithm Core (preserve 1:1)
- [List core modules and their roles]
- [List key hyperparameters and defaults]
- [List special implementation details]

### Engineering Changes (planned)
- [Interfaces to adjust]
- [Framework adaptation strategy]
- [I/O format changes]

### Risks
- [Changes that might affect algorithm correctness]
```

### Phase 3: Implementation

Sequence:

1. Copy algorithm core verbatim (line-by-line check).
2. Add engineering adapters (wrapper/adapter pattern).
3. Add verification hooks to confirm identical outputs.

### Phase 4: Self-Check

Run this checklist before delivering:

```markdown
## Self-Check Report

### Algorithm Integrity
- [ ] Layer counts match the original
- [ ] Per-layer parameter counts match
- [ ] Loss and math formulas unchanged
- [ ] Default hyperparameters unchanged
- [ ] Special init/regularization preserved

### Engineering Changes
- [Change 1]: [Reason]
- [Change 2]: [Reason]

### Missing Items (if any)
- [Item]: [Reason] [Impact on algorithm: yes/no]
```

## How to Modify Engineering Safely

### ✅ Correct: Wrapper Pattern

```python
# Original model is untouched
class OriginalModel(nn.Module):
    # Copy original implementation verbatim
    ...

# Wrapper adapts interfaces
class ModelWrapper:
    def __init__(self):
        self.model = OriginalModel()  # original model
    
    def predict(self, input_dict):
        # Convert formats in the wrapper
        x = self._preprocess(input_dict)
        out = self.model(x)
        return self._postprocess(out)
```

### ❌ Wrong: Editing Algorithm Code

```python
# Do NOT do this
class OriginalModel(nn.Module):
    def forward(self, x):
        # Original: complex multi-head attention
        # Simplified to:
        return F.linear(x, self.weight)  # Algorithm is broken
```

## Framework Migration Guide

When porting from framework A to B:

1. Keep the computation graph equivalent: every op must have a counterpart.
2. Verify numerical parity: same input -> same output (float tolerance < 1e-5).
3. Preserve original naming to ease diffing.

```python
# Parity check example
def verify_equivalence(model_a, model_b, test_input):
    out_a = model_a(test_input)
    out_b = model_b(test_input)
    assert torch.allclose(out_a, out_b, atol=1e-5), "Implementations are not equivalent!"
```

## Common Pitfalls

| Pitfall | Symptom | Risk |
|---------|---------|------|
| "Looks similar" | Swap in a similar but different implementation | Behavior drifts from paper |
| "Probably not important" | Remove code that seems redundant | Hidden critical detail lost |
| "Temporary simplification" | Use placeholder or simplified logic | Likely never replaced; breaks fidelity |
| "Defaults are fine" | Skip checking original hyperparams | Results become unreproducible |

## Special Situations

### Missing Source

```
1. Tell the user which parts are missing.
2. Label any inferred code as [INFERRED].
3. Offer multiple plausible implementations.
4. Do not "fill in" on your own without sign-off.
```

### Performance Optimizations

```
1. Finish an equivalent implementation first.
2. Verify correctness before optimizing.
3. Keep mathematical equivalence when optimizing.
4. Record every optimization and its rationale.
```

### Dependency Version Issues

```
1. Record original dependency versions.
2. Check API changes carefully.
3. Use compatibility wrappers instead of altering algorithm code.
```
