# Distributed Training Patterns

## Core Principle

**Abstract distribution strategy from training logic.**

Training code should not care whether it runs on 1 GPU or 1000.

---

## Strategy Abstraction

```python
# src/distributed/strategy.py
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import torch
import torch.nn as nn

class DistributedStrategy(ABC):
    """Abstract distributed training strategy."""
    
    @classmethod
    def from_config(cls, config) -> "DistributedStrategy":
        """Factory method to create strategy from config."""
        strategy_name = config.distributed.strategy
        
        if strategy_name == "fsdp":
            return FSDPStrategy(config)
        elif strategy_name == "deepspeed":
            return DeepSpeedStrategy(config)
        elif strategy_name == "ddp":
            return DDPStrategy(config)
        else:
            return SingleGPUStrategy(config)
    
    @property
    @abstractmethod
    def world_size(self) -> int:
        """Total number of processes."""
        pass
    
    @property
    @abstractmethod
    def rank(self) -> int:
        """Current process rank."""
        pass
    
    @property
    @abstractmethod
    def local_rank(self) -> int:
        """Local rank within node."""
        pass
    
    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process."""
        return self.rank == 0
    
    @abstractmethod
    def setup(self):
        """Initialize distributed environment."""
        pass
    
    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training."""
        pass
    
    @abstractmethod
    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient handling."""
        pass
    
    @abstractmethod
    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        """Clip gradients."""
        pass
    
    @abstractmethod
    def save_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        **kwargs
    ):
        """Save checkpoint with sharding if needed."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: Path) -> dict:
        """Load checkpoint."""
        pass
    
    def to_device(self, batch: Any) -> Any:
        """Move batch to appropriate device."""
        if hasattr(batch, '_fields'):  # namedtuple
            return type(batch)(*[self.to_device(x) for x in batch])
        elif isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(f"cuda:{self.local_rank}")
        return batch
```

---

## FSDP Strategy

```python
# src/distributed/fsdp.py
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist

class FSDPStrategy(DistributedStrategy):
    """Fully Sharded Data Parallel strategy."""
    
    def __init__(self, config):
        self.config = config
        self._rank = None
        self._world_size = None
        self._local_rank = None
    
    @property
    def world_size(self) -> int:
        return self._world_size
    
    @property
    def rank(self) -> int:
        return self._rank
    
    @property
    def local_rank(self) -> int:
        return self._local_rank
    
    def setup(self):
        """Initialize FSDP distributed environment."""
        dist.init_process_group(backend="nccl")
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self._local_rank)
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with FSDP."""
        # Get transformer layer class for wrapping
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={LlamaDecoderLayer},
        )
        
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        
        return FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=self._local_rank,
            limit_all_gathers=True,
        )
    
    def backward(self, loss: torch.Tensor):
        loss.backward()
    
    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        model.clip_grad_norm_(max_norm)
    
    def save_checkpoint(self, path: Path, model: nn.Module, optimizer, **kwargs):
        """Save FSDP checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Full state dict on rank 0
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
        ):
            state_dict = model.state_dict()
            if self.is_main_process:
                torch.save(state_dict, path / "model.pt")
        
        # Optimizer state (sharded)
        optim_state = FSDP.optim_state_dict(model, optimizer)
        torch.save(optim_state, path / f"optim_rank{self.rank}.pt")
        
        # Training state on rank 0
        if self.is_main_process:
            torch.save(kwargs, path / "training_state.pt")
        
        dist.barrier()
    
    def load_checkpoint(self, path: Path) -> dict:
        """Load FSDP checkpoint."""
        # Model weights loaded before FSDP wrapping
        # Optimizer state loaded after
        state = {}
        if (path / "training_state.pt").exists():
            state = torch.load(path / "training_state.pt")
        return state
```

---

## DeepSpeed Strategy

```python
# src/distributed/deepspeed.py
import deepspeed

class DeepSpeedStrategy(DistributedStrategy):
    """DeepSpeed ZeRO strategy."""
    
    def __init__(self, config):
        self.config = config
        self.ds_config = self._build_ds_config()
    
    def _build_ds_config(self) -> dict:
        """Build DeepSpeed config from our config."""
        return {
            "train_batch_size": self.config.training.batch_size * self.world_size,
            "gradient_accumulation_steps": self.config.training.gradient_accumulation,
            "fp16": {"enabled": self.config.training.mixed_precision == "fp16"},
            "bf16": {"enabled": self.config.training.mixed_precision == "bf16"},
            "zero_optimization": {
                "stage": self.config.distributed.zero_stage,
                "offload_optimizer": {"device": "cpu"} if self.config.distributed.offload else {},
            },
            "gradient_clipping": self.config.training.max_grad_norm,
        }
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=self.ds_config,
            model_parameters=model.parameters(),
        )
        self._optimizer = optimizer
        self._scheduler = scheduler
        return model_engine
    
    def backward(self, loss: torch.Tensor):
        self.model_engine.backward(loss)
    
    def save_checkpoint(self, path: Path, model: nn.Module, **kwargs):
        model.save_checkpoint(path, client_state=kwargs)
    
    def load_checkpoint(self, path: Path) -> dict:
        _, client_state = self.model_engine.load_checkpoint(path)
        return client_state
```

---

## Single GPU Strategy (Development)

```python
# src/distributed/single.py
class SingleGPUStrategy(DistributedStrategy):
    """Single GPU strategy for development."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    @property
    def world_size(self) -> int:
        return 1
    
    @property
    def rank(self) -> int:
        return 0
    
    @property
    def local_rank(self) -> int:
        return 0
    
    def setup(self):
        pass
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.to(self.device)
    
    def backward(self, loss: torch.Tensor):
        loss.backward()
    
    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    def save_checkpoint(self, path: Path, model: nn.Module, optimizer, **kwargs):
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            **kwargs
        }, path / "checkpoint.pt")
    
    def load_checkpoint(self, path: Path) -> dict:
        return torch.load(path / "checkpoint.pt")
```

---

## Checkpoint Utilities

```python
# src/distributed/checkpoint.py
from pathlib import Path
from typing import Optional
import torch
import torch.distributed as dist

def find_latest_checkpoint(experiment_dir: Path) -> Optional[Path]:
    """Find latest checkpoint in experiment directory."""
    ckpt_dir = experiment_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    
    checkpoints = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    return checkpoints[0] if checkpoints else None


def auto_resume(config, trainer):
    """Auto-resume from latest checkpoint if exists."""
    latest = find_latest_checkpoint(config.experiment.output_dir)
    if latest:
        trainer.load_checkpoint(latest)
        return True
    return False


class CheckpointManager:
    """Manage checkpoint saving with rotation."""
    
    def __init__(self, output_dir: Path, keep_last: int = 3):
        self.output_dir = output_dir / "checkpoints"
        self.keep_last = keep_last
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, strategy: DistributedStrategy, step: int, **kwargs):
        """Save checkpoint and rotate old ones."""
        path = self.output_dir / f"step_{step}"
        strategy.save_checkpoint(path, **kwargs)
        self._rotate()
    
    def _rotate(self):
        """Keep only last N checkpoints."""
        checkpoints = sorted(
            self.output_dir.iterdir(),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for old_ckpt in checkpoints[self.keep_last:]:
            if old_ckpt.name != "final":
                shutil.rmtree(old_ckpt)
```

---

## Launch Scripts

```python
# src/distributed/launcher.py
import subprocess
import os

def launch_distributed(
    script: str,
    config: str,
    num_nodes: int = 1,
    gpus_per_node: int = 8,
    master_addr: str = "localhost",
    master_port: int = 29500,
):
    """Launch distributed training with torchrun."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus_per_node}",
        f"--nnodes={num_nodes}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        script,
        config,
    ]
    subprocess.run(cmd, check=True)
```

```bash
# scripts/launch.sh
#!/bin/bash
# Multi-node launch script

export MASTER_ADDR=${MASTER_ADDR:-"node0"}
export MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
    --nproc_per_node=8 \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py $@
```

---

## Best Practices

### 1. Always Use Strategy Abstraction
```python
# Good
strategy = DistributedStrategy.from_config(config)
model = strategy.wrap_model(model)

# Bad - hardcoded FSDP
model = FSDP(model, ...)  # Can't switch strategies
```

### 2. Guard Rank-Specific Operations
```python
# Logging only on main process
if strategy.is_main_process:
    wandb.log(metrics)

# Barrier before checkpoint loading
dist.barrier()
```

### 3. Test Locally First
```yaml
# configs/base/distributed.yaml
distributed:
  strategy: single  # For development

# configs/experiments/prod.yaml
distributed:
  strategy: fsdp    # For production
```

### 4. Handle Gradient Accumulation
```python
# Effective batch size = batch_size * grad_accum * world_size
effective_batch_size = (
    config.training.batch_size 
    * config.training.gradient_accumulation 
    * strategy.world_size
)
```
