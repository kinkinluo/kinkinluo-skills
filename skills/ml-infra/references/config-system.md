# Configuration System

## Core Principle

**Configuration drives experiments, not code changes.**

One codebase supports many experiments through config inheritance and composition.

---

## Config Inheritance Pattern

```yaml
# configs/base/training.yaml
training:
  batch_size: 32
  learning_rate: 1e-4
  warmup_steps: 100
  total_steps: 10000
  gradient_accumulation: 1
  mixed_precision: bf16
  
optimizer:
  name: adamw
  betas: [0.9, 0.999]
  weight_decay: 0.01

logging:
  log_interval: 10
  eval_interval: 500
  save_interval: 1000
```

```yaml
# configs/algorithms/dpo.yaml
_base_:
  - ../base/training.yaml

algorithm:
  name: dpo
  
dpo:
  beta: 0.1
  label_smoothing: 0.0
  loss_type: sigmoid  # sigmoid | hinge | ipo
  reference_free: false
```

```yaml
# configs/experiments/exp001_dpo_7b.yaml
_base_:
  - ../algorithms/dpo.yaml
  - ../base/model.yaml

experiment:
  name: exp001_dpo_7b
  seed: 42
  
model:
  name: llama-7b
  path: /models/llama-7b-base
  
training:
  total_steps: 5000      # Override base
  batch_size: 16         # Override base
  
dpo:
  beta: 0.05             # Override algorithm default
```

---

## Config Implementation

```python
# src/core/config.py
from pathlib import Path
from typing import Any
import yaml
from dataclasses import dataclass, field

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def resolve_inheritance(config_path: Path) -> dict:
    """Resolve _base_ inheritance chain."""
    config = load_yaml(config_path)
    
    if "_base_" not in config:
        return config
    
    bases = config.pop("_base_")
    if isinstance(bases, str):
        bases = [bases]
    
    # Resolve each base relative to current config
    merged = {}
    for base_path in bases:
        base_full_path = config_path.parent / base_path
        base_config = resolve_inheritance(base_full_path)
        merged = deep_merge(merged, base_config)
    
    # Override with current config
    return deep_merge(merged, config)


@dataclass
class Config:
    """Type-safe config wrapper."""
    _data: dict = field(repr=False)
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        value = self._data.get(name)
        if isinstance(value, dict):
            return Config(_data=value)
        return value
    
    def to_dict(self) -> dict:
        return self._data
    
    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        path = Path(path)
        data = resolve_inheritance(path)
        return cls(_data=data)
    
    @classmethod
    def from_cli(cls) -> "Config":
        """Parse config path and overrides from CLI."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("config", type=Path)
        parser.add_argument("--override", "-o", nargs="*", default=[])
        args = parser.parse_args()
        
        config = cls.from_file(args.config)
        
        # Apply CLI overrides: --override training.lr=1e-5
        for override in args.override:
            key, value = override.split("=")
            config.set_nested(key, yaml.safe_load(value))
        
        return config
    
    def set_nested(self, key: str, value: Any):
        """Set nested key like 'training.lr'."""
        keys = key.split(".")
        data = self._data
        for k in keys[:-1]:
            data = data.setdefault(k, {})
        data[keys[-1]] = value
```

---

## Usage Patterns

### Training Script
```python
# scripts/train.py
from src.core import Config, ALGORITHMS

def main():
    config = Config.from_cli()
    
    # Log full resolved config for reproducibility
    config.save(config.experiment.output_dir / "config.yaml")
    
    trainer = ALGORITHMS.build(config.algorithm.name, config=config)
    trainer.train()

# Run:
# python scripts/train.py configs/experiments/exp001.yaml
# python scripts/train.py configs/experiments/exp001.yaml -o training.lr=5e-5
```

### Accessing Config Values
```python
# Type-safe nested access
lr = config.training.learning_rate
model_name = config.model.name
beta = config.dpo.beta

# Check existence
if config.distributed:
    setup_distributed(config.distributed)

# Get with default
batch_size = config.training.batch_size or 32
```

### Config Validation
```python
# src/core/config.py (addition)
from pydantic import BaseModel, validator

class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    total_steps: int
    warmup_steps: int = 0
    
    @validator("learning_rate")
    def lr_positive(cls, v):
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        return v

class DPOConfig(BaseModel):
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    
    @validator("loss_type")
    def valid_loss_type(cls, v):
        valid = ["sigmoid", "hinge", "ipo"]
        if v not in valid:
            raise ValueError(f"loss_type must be one of {valid}")
        return v

# Validate on load
def validate_config(config: Config) -> Config:
    TrainingConfig(**config.training.to_dict())
    if config.algorithm.name == "dpo":
        DPOConfig(**config.dpo.to_dict())
    return config
```

---

## Best Practices

### 1. Organize by Abstraction Level
```
configs/
├── base/           # Lowest level, most reusable
├── algorithms/     # Algorithm-specific, inherits base
├── models/         # Model-specific configs
└── experiments/    # Highest level, inherits all
```

### 2. Keep Base Configs Minimal
```yaml
# Good: only essential shared settings
training:
  mixed_precision: bf16
  gradient_checkpointing: true

# Bad: too specific for base
training:
  batch_size: 32          # Varies by experiment
  total_steps: 10000      # Varies by experiment
```

### 3. Document Config Options
```yaml
# configs/algorithms/dpo.yaml
dpo:
  # KL penalty coefficient. Higher = closer to reference model.
  # Typical range: 0.01 - 0.5
  beta: 0.1
  
  # Label smoothing for preference pairs
  # 0.0 = no smoothing, 0.1 = common choice
  label_smoothing: 0.0
  
  # Loss function variant
  # Options: sigmoid (original DPO), hinge, ipo
  loss_type: sigmoid
```

### 4. Freeze Configs for Experiments
```python
# At experiment start, save resolved config
def setup_experiment(config: Config) -> Path:
    output_dir = Path(f"experiments/{config.experiment.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full resolved config (no inheritance)
    full_config = config.to_dict()
    full_config["_resolved_at"] = datetime.now().isoformat()
    full_config["_git_hash"] = get_git_hash()
    
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(full_config, f, default_flow_style=False)
    
    return output_dir
```

### 5. Support Environment Variables
```yaml
# configs/base/paths.yaml
paths:
  data_root: ${DATA_ROOT:/data}
  model_cache: ${HF_HOME:/models}
  output_root: ${OUTPUT_ROOT:./experiments}
```

```python
import os
import re

def expand_env_vars(config: dict) -> dict:
    """Expand ${VAR:default} patterns."""
    pattern = re.compile(r'\$\{(\w+)(?::([^}]*))?\}')
    
    def expand(value):
        if isinstance(value, str):
            def replace(match):
                var, default = match.groups()
                return os.environ.get(var, default or "")
            return pattern.sub(replace, value)
        elif isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand(v) for v in value]
        return value
    
    return expand(config)
```
