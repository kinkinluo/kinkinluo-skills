# Experiment Management

## Core Principle

**Every experiment must be reproducible.**

This requires tracking: code version, config, environment, and random seeds.

---

## Experiment Structure

```
experiments/
└── exp001_dpo_7b_beta0.1/
    ├── config.yaml          # Frozen full config (no inheritance)
    ├── metadata.json        # Git hash, timestamp, environment
    ├── checkpoints/
    │   ├── step_1000/
    │   ├── step_2000/
    │   └── final/
    ├── logs/
    │   ├── train.log        # Full training log
    │   ├── metrics.jsonl    # Structured metrics
    │   └── tensorboard/
    └── outputs/
        ├── samples/         # Generated samples
        └── eval/            # Evaluation results
```

---

## Experiment Setup

```python
# src/core/experiment.py
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import subprocess
import yaml
import os

@dataclass
class ExperimentMetadata:
    name: str
    started_at: str
    git_hash: str
    git_branch: str
    git_dirty: bool
    python_version: str
    torch_version: str
    cuda_version: str
    hostname: str
    command: str

def get_git_info() -> dict:
    """Get current git state."""
    try:
        return {
            "git_hash": subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode().strip(),
            "git_branch": subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            ).decode().strip(),
            "git_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"]
            ).decode().strip()),
        }
    except:
        return {"git_hash": "unknown", "git_branch": "unknown", "git_dirty": True}

def setup_experiment(config) -> Path:
    """Initialize experiment directory and save metadata."""
    import torch
    import sys
    import socket
    
    # Create experiment directory
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "outputs").mkdir(exist_ok=True)
    
    # Save frozen config (fully resolved, no inheritance)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    # Save metadata
    git_info = get_git_info()
    metadata = ExperimentMetadata(
        name=config.experiment.name,
        started_at=datetime.now().isoformat(),
        python_version=sys.version,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "N/A",
        hostname=socket.gethostname(),
        command=" ".join(sys.argv),
        **git_info,
    )
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(asdict(metadata), f, indent=2)
    
    # Warn if git is dirty
    if git_info["git_dirty"]:
        print("⚠️  WARNING: Git working directory has uncommitted changes!")
    
    return output_dir
```

---

## Reproducibility

### Seed Everything

```python
# src/core/reproducibility.py
import random
import numpy as np
import torch

def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For fully deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_rng_state() -> dict:
    """Capture current RNG state for checkpointing."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }

def set_rng_state(state: dict):
    """Restore RNG state from checkpoint."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
```

### Environment Tracking

```python
# src/core/environment.py
import subprocess

def save_environment(output_dir: Path):
    """Save environment for reproducibility."""
    # pip freeze
    pip_freeze = subprocess.check_output(["pip", "freeze"]).decode()
    (output_dir / "requirements.txt").write_text(pip_freeze)
    
    # conda (if applicable)
    try:
        conda_list = subprocess.check_output(["conda", "list", "--export"]).decode()
        (output_dir / "conda_env.txt").write_text(conda_list)
    except:
        pass

def create_reproduction_script(output_dir: Path, config_path: Path):
    """Generate script to reproduce experiment."""
    script = f'''#!/bin/bash
# Reproduction script for {output_dir.name}
# Generated: {datetime.now().isoformat()}

# 1. Checkout correct commit
git checkout {get_git_info()["git_hash"]}

# 2. Install dependencies
pip install -r {output_dir}/requirements.txt

# 3. Run training
python scripts/train.py {config_path}
'''
    (output_dir / "reproduce.sh").write_text(script)
```

---

## Logging

### Structured Logger

```python
# src/core/logging.py
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

class Logger:
    """Unified logger for training."""
    
    def __init__(self, config, rank: int = 0):
        self.config = config
        self.rank = rank
        self.is_main = rank == 0
        
        self.output_dir = Path(config.experiment.output_dir) / "logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logger
        self._setup_file_logger()
        
        # Setup experiment tracker (wandb, etc.)
        if self.is_main:
            self._setup_tracker()
    
    def _setup_file_logger(self):
        self.file_logger = logging.getLogger(f"train_{self.rank}")
        self.file_logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.output_dir / f"train_rank{self.rank}.log")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        self.file_logger.addHandler(handler)
        
        # Metrics file (JSONL for easy parsing)
        self.metrics_file = open(self.output_dir / "metrics.jsonl", "a")
    
    def _setup_tracker(self):
        """Setup experiment tracker (wandb, etc.)"""
        tracker = self.config.logging.get("tracker", "none")
        
        if tracker == "wandb":
            import wandb
            wandb.init(
                project=self.config.logging.project,
                name=self.config.experiment.name,
                config=self.config.to_dict(),
            )
            self.wandb = wandb
        elif tracker == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.output_dir / "tensorboard")
    
    def info(self, message: str):
        """Log info message."""
        if self.is_main:
            self.file_logger.info(message)
            print(message)
    
    def log_metrics(
        self, 
        metrics: dict, 
        step: int, 
        prefix: str = "train"
    ):
        """Log metrics to all destinations."""
        # Add metadata
        record = {
            "step": step,
            "prefix": prefix,
            "timestamp": datetime.now().isoformat(),
            **{f"{prefix}/{k}": v for k, v in metrics.items()}
        }
        
        # File (all ranks)
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()
        
        # Tracker (main only)
        if self.is_main:
            if hasattr(self, "wandb"):
                self.wandb.log(record, step=step)
            if hasattr(self, "tb_writer"):
                for k, v in metrics.items():
                    self.tb_writer.add_scalar(f"{prefix}/{k}", v, step)
    
    def log_artifact(self, path: Path, name: str, type: str = "model"):
        """Log artifact (checkpoint, etc.) to tracker."""
        if self.is_main and hasattr(self, "wandb"):
            artifact = self.wandb.Artifact(name, type=type)
            artifact.add_dir(str(path))
            self.wandb.log_artifact(artifact)
    
    def close(self):
        """Cleanup."""
        self.metrics_file.close()
        if hasattr(self, "wandb"):
            self.wandb.finish()
        if hasattr(self, "tb_writer"):
            self.tb_writer.close()
```

### Config for Logging

```yaml
# configs/base/logging.yaml
logging:
  tracker: wandb           # wandb | tensorboard | none
  project: llm-training    # wandb project name
  log_interval: 10         # Log every N steps
  eval_interval: 500       # Evaluate every N steps
  save_interval: 1000      # Checkpoint every N steps
```

---

## Experiment Naming

```python
# src/core/naming.py
from datetime import datetime

def generate_experiment_name(config) -> str:
    """Generate descriptive experiment name."""
    parts = [
        config.algorithm.name,                    # dpo
        config.model.name.split("/")[-1],         # llama-7b
    ]
    
    # Add key hyperparameters
    if config.algorithm.name == "dpo":
        parts.append(f"beta{config.dpo.beta}")
    elif config.algorithm.name == "ppo":
        parts.append(f"clip{config.ppo.clip_ratio}")
    
    # Add timestamp
    parts.append(datetime.now().strftime("%m%d_%H%M"))
    
    return "_".join(parts)

# Example: dpo_llama-7b_beta0.1_0115_1430
```

---

## Analysis Tools

```python
# tools/analyze_experiment.py
"""Analyze completed experiment."""
import json
from pathlib import Path
import pandas as pd

def load_metrics(experiment_dir: Path) -> pd.DataFrame:
    """Load metrics from JSONL file."""
    metrics_file = experiment_dir / "logs" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_file.read_text().splitlines()]
    return pd.DataFrame(records)

def summarize_experiment(experiment_dir: Path) -> dict:
    """Generate experiment summary."""
    df = load_metrics(experiment_dir)
    
    # Load config
    config = yaml.safe_load((experiment_dir / "config.yaml").read_text())
    
    # Final metrics
    train_metrics = df[df["prefix"] == "train"].iloc[-1].to_dict()
    eval_metrics = df[df["prefix"] == "eval"].iloc[-1].to_dict()
    
    return {
        "name": config["experiment"]["name"],
        "algorithm": config["algorithm"]["name"],
        "total_steps": train_metrics["step"],
        "final_loss": train_metrics.get("train/loss"),
        "final_eval": eval_metrics,
    }

def compare_experiments(experiment_dirs: list[Path]) -> pd.DataFrame:
    """Compare multiple experiments."""
    summaries = [summarize_experiment(d) for d in experiment_dirs]
    return pd.DataFrame(summaries)
```

---

## Best Practices

### 1. Never Modify Config After Start
```python
# Good
config = Config.from_cli()
setup_experiment(config)  # Saves frozen config
trainer.train()

# Bad
trainer.train()
config.training.lr = new_lr  # Config already saved!
```

### 2. Include All Hyperparameters
```yaml
# Good - explicit
dpo:
  beta: 0.1
  label_smoothing: 0.0  # Even if default

# Bad - relies on defaults
dpo:
  beta: 0.1
  # label_smoothing not specified - what was used?
```

### 3. Track Failed Experiments
```python
def train():
    try:
        trainer.train()
    except Exception as e:
        # Save failure info
        with open(output_dir / "FAILED.txt", "w") as f:
            f.write(f"Failed at step {trainer.global_step}\n")
            f.write(traceback.format_exc())
        raise
```

### 4. Use Descriptive Names
```
# Good
exp001_dpo_llama7b_beta0.1_lr1e-5

# Bad
test1
experiment_final_v2_FINAL
```
