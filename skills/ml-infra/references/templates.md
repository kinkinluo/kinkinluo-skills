# ML Project Templates

## Table of Contents
1. [LLM Post-Training (RLHF/DPO)](#llm-post-training)
2. [LLM Pre-Training](#llm-pre-training)
3. [Vision Research](#vision-research)
4. [General ML Research](#general-research)
5. [Production ML Pipeline](#production-ml)

---

## LLM Post-Training (RLHF/DPO) {#llm-post-training}

```
llm-post-training/
├── configs/
│   ├── base/
│   │   ├── model.yaml              # Model architecture
│   │   ├── training.yaml           # General training
│   │   ├── distributed.yaml        # FSDP/DeepSpeed
│   │   └── generation.yaml         # Sampling params
│   ├── algorithms/
│   │   ├── sft.yaml
│   │   ├── ppo.yaml
│   │   ├── dpo.yaml
│   │   └── grpo.yaml
│   └── experiments/
│       └── exp001_dpo_7b.yaml
│
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── registry.py             # Component registration
│   │   ├── config.py               # Config parser with inheritance
│   │   ├── logging.py              # Wandb/tensorboard wrapper
│   │   └── types.py                # Shared type definitions
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── policy.py               # Policy model wrapper
│   │   ├── reference.py            # Frozen reference model
│   │   ├── reward.py               # Reward model
│   │   ├── value.py                # Value head for PPO
│   │   └── utils.py                # Model utilities
│   │
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseTrainer ABC
│   │   ├── sft/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py
│   │   │   └── loss.py
│   │   ├── ppo/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py
│   │   │   ├── loss.py
│   │   │   └── buffer.py           # Experience buffer
│   │   ├── dpo/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py
│   │   │   └── loss.py
│   │   └── grpo/
│   │       └── ...
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets/
│   │   │   ├── prompt.py           # Prompt-only dataset
│   │   │   ├── preference.py       # Preference pairs
│   │   │   └── sft.py              # SFT format
│   │   ├── collators.py
│   │   ├── samplers.py             # Distributed samplers
│   │   └── preprocessing.py
│   │
│   ├── distributed/
│   │   ├── __init__.py
│   │   ├── strategy.py             # FSDP/DeepSpeed abstraction
│   │   ├── checkpoint.py           # Save/load with sharding
│   │   ├── launcher.py             # torchrun wrapper
│   │   └── utils.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmarks/
│   │   │   ├── mmlu.py
│   │   │   ├── humaneval.py
│   │   │   └── custom.py
│   │   ├── judges/
│   │   │   ├── reward_judge.py
│   │   │   └── llm_judge.py
│   │   └── metrics.py
│   │
│   └── generation/
│       ├── __init__.py
│       ├── sampler.py              # Generation with policy
│       └── batch_inference.py      # Efficient batch generation
│
├── scripts/
│   ├── train.py                    # Unified training entry
│   ├── evaluate.py
│   ├── generate.py
│   └── serve.py                    # vLLM serving
│
├── tools/
│   ├── merge_lora.py
│   ├── convert_checkpoint.py
│   ├── analyze_rewards.py
│   └── visualize_training.py
│
├── experiments/                    # gitignore
│   └── exp001/
│       ├── config.yaml             # Frozen config
│       ├── checkpoints/
│       ├── logs/
│       └── outputs/
│
├── pyproject.toml
└── README.md
```

**Entry point pattern:**
```python
# scripts/train.py
from src.core import Config, Registry
from src.algorithms import ALGORITHMS

def main():
    config = Config.from_cli()  # Load with inheritance
    
    trainer = ALGORITHMS.build(
        config.algorithm.name,
        config=config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
```

---

## LLM Pre-Training {#llm-pre-training}

```
llm-pretrain/
├── configs/
│   ├── base/
│   │   ├── model/
│   │   │   ├── llama_7b.yaml
│   │   │   ├── llama_13b.yaml
│   │   │   └── llama_70b.yaml
│   │   ├── training.yaml
│   │   ├── data.yaml
│   │   └── distributed.yaml
│   └── experiments/
│
├── src/
│   ├── core/
│   │   └── ...
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llama/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── attention.py
│   │   │   └── config.py
│   │   └── components/
│   │       ├── embeddings.py
│   │       ├── normalization.py
│   │       └── activations.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   ├── streaming.py            # Streaming dataset
│   │   ├── packing.py              # Sequence packing
│   │   └── mixture.py              # Data mixture
│   │
│   ├── distributed/
│   │   ├── __init__.py
│   │   ├── tensor_parallel.py
│   │   ├── pipeline_parallel.py
│   │   ├── fsdp.py
│   │   └── checkpoint.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── optimizer.py            # AdamW, schedulers
│   │   └── loss.py
│   │
│   └── evaluation/
│       └── perplexity.py
│
├── scripts/
│   ├── train.py
│   ├── tokenize_data.py
│   └── validate_checkpoint.py
│
└── data/                           # Data processing outputs
    ├── tokenized/
    └── packed/
```

---

## Vision Research {#vision-research}

```
vision-research/
├── configs/
│   ├── base/
│   │   ├── model.yaml
│   │   ├── training.yaml
│   │   └── augmentation.yaml
│   ├── datasets/
│   │   ├── imagenet.yaml
│   │   ├── coco.yaml
│   │   └── custom.yaml
│   ├── models/
│   │   ├── resnet.yaml
│   │   ├── vit.yaml
│   │   └── custom.yaml
│   └── experiments/
│
├── src/
│   ├── core/
│   │   └── ...
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones/
│   │   │   ├── resnet.py
│   │   │   ├── vit.py
│   │   │   └── convnext.py
│   │   ├── heads/
│   │   │   ├── classification.py
│   │   │   ├── detection.py
│   │   │   └── segmentation.py
│   │   ├── necks/
│   │   │   └── fpn.py
│   │   └── losses/
│   │       ├── cross_entropy.py
│   │       ├── focal.py
│   │       └── contrastive.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets/
│   │   │   ├── classification.py
│   │   │   └── detection.py
│   │   ├── transforms/
│   │   │   ├── geometric.py
│   │   │   ├── color.py
│   │   │   └── mixup.py
│   │   └── samplers.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── hooks.py                # Training callbacks
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── classification.py
│       └── detection.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
│
└── notebooks/
    └── exploration.ipynb
```

---

## General ML Research {#general-research}

Minimal structure for quick experiments:

```
research-project/
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── exp001.yaml
│       └── exp002.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Simple config loader
│   ├── model.py                    # Model definition
│   ├── data.py                     # Dataset and transforms
│   ├── trainer.py                  # Training loop
│   ├── evaluate.py                 # Evaluation
│   └── utils.py
│
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│   ├── exploration.ipynb
│   └── analysis.ipynb
│
├── experiments/                    # gitignore
└── pyproject.toml
```

**When to scale up:**
- Multiple model architectures → split `model.py` into `models/`
- Multiple algorithms → create `algorithms/` with registry
- Distributed training needed → add `distributed/`
- Complex data pipeline → create `data/` package

---

## Production ML Pipeline {#production-ml}

```
ml-pipeline/
├── configs/
│   ├── training/
│   ├── serving/
│   └── monitoring/
│
├── src/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── features.py
│   │   └── model.py
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   ├── preprocessing.py
│   │   └── postprocessing.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── drift.py
│   │
│   └── common/
│       ├── __init__.py
│       ├── features.py             # Shared feature definitions
│       └── schemas.py              # Input/output schemas
│
├── pipelines/                      # Orchestration (Airflow/Kubeflow)
│   ├── training_pipeline.py
│   └── inference_pipeline.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── deployment/
│   ├── docker/
│   └── kubernetes/
│
└── scripts/
    ├── train.py
    ├── serve.py
    └── evaluate.py
```
