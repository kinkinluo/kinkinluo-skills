---
name: ml-infra
description: |
  Engineering patterns for ML/DL research and LLM training infrastructure. Use this skill when: (1) Building training pipelines (SFT, RLHF, DPO, etc.), (2) Creating ML experiment infrastructure, (3) Designing distributed training systems, (4) Research projects requiring reproducibility, (5) Any deep learning project beyond simple scripts. Complements engineering-code skill with ML-specific patterns: config-driven architecture, component registry, experiment management, and stable/experimental layer separation.
---

# ML Infrastructure

Engineering patterns for ML/DL research projects that balance reproducibility with iteration speed.

## Core Philosophy

**Research projects differ from typical software:**
- Experiments change frequently, infrastructure should be stable
- Configuration drives behavior, not code changes
- Reproducibility is non-negotiable
- Same codebase runs many experiments

**Solution: Separate stable infrastructure from experimental code.**

## Project Structure

```
project/
├── configs/                    # Configuration center
│   ├── base/                   # Base configurations
│   ├── algorithms/             # Algorithm-specific
│   └── experiments/            # Experiment overrides
│
├── src/
│   ├── core/                   # Stable infrastructure
│   ├── models/                 # Model definitions
│   ├── algorithms/             # Training algorithms
│   ├── data/                   # Data pipeline
│   ├── distributed/            # Distributed training
│   ├── evaluation/             # Evaluation pipeline
│   └── generation/             # Inference/sampling
│
├── scripts/                    # Entry points
├── tools/                      # Utilities
├── notebooks/                  # Exploration (gitignore outputs)
└── experiments/                # Outputs (gitignore)
```

See `references/templates.md` for complete templates by project type.

## Config-Driven Architecture

**Principle: One codebase, many experiments via configuration.**

```yaml
# configs/experiments/exp001.yaml
_base_:
  - ../base/model.yaml
  - ../base/training.yaml
  - ../algorithms/ppo.yaml

model:
  name: llama-7b
  
training:
  total_steps: 10000
  
ppo:
  clip_ratio: 0.2  # Override base
```

See `references/config-system.md` for inheritance patterns and best practices.

## Component Registry

**Principle: Instantiate components from config strings.**

```python
@ALGORITHMS.register("ppo")
class PPOTrainer(BaseTrainer):
    ...

# Usage
trainer = ALGORITHMS.build(config.algorithm)
```

See `references/registry.md` for implementation patterns.

## Stable vs Experimental Layers

| Stable (strict engineering) | Experimental (flexible) |
|----------------------------|------------------------|
| `core/` - registry, config, logging | `algorithms/` - new methods |
| `distributed/` - FSDP, checkpointing | `models/` - new architectures |
| `data/` - loaders, samplers | Loss functions, metrics |
| Training loop skeleton | Training step internals |

**Rule: Experimental code plugs into stable interfaces.**

## Abstract Base Classes

Every algorithm implements standard interfaces:

```python
class BaseTrainer(ABC):
    @abstractmethod
    def compute_loss(self, batch: Batch) -> LossOutput: ...
    
    @abstractmethod
    def training_step(self, batch: Batch) -> StepOutput: ...
    
    # Shared implementation
    def train(self): ...  # Training loop
    def save_checkpoint(self): ...
    def load_checkpoint(self): ...
```

See `references/abstractions.md` for complete interface definitions.

## File Size Guidelines

| File Type | Max Lines | Notes |
|-----------|-----------|-------|
| Config files | 100 | Use inheritance for longer |
| Core infra | 200 | Stable, well-documented |
| Algorithm impl | 300 | Complex logic acceptable |
| Model definition | 400 | Large models need space |
| Training script | 150 | Thin orchestration layer |

## Experiment Management

Every experiment produces:
```
experiments/exp001/
├── config.yaml      # Frozen full config
├── checkpoints/
├── logs/
│   ├── metrics.jsonl
│   └── tensorboard/
└── outputs/         # Generated samples, eval results
```

**Rules:**
- Never modify code to run experiments, only configs
- Commit config to git, gitignore outputs
- Log full resolved config at experiment start

## Workflow

1. **New project**: Read `references/templates.md` → Initialize structure
2. **New algorithm**: Inherit base class → Register → Create config
3. **New experiment**: Create config inheriting base → Run
4. **Iteration**: Modify experimental layer only, stable layer unchanged

## References

- `references/templates.md` - Project templates (LLM training, vision, research)
- `references/config-system.md` - Configuration inheritance patterns
- `references/registry.md` - Component registry implementation
- `references/abstractions.md` - Base classes and interfaces
- `references/distributed.md` - Distributed training patterns
- `references/experiment.md` - Experiment tracking and reproducibility
