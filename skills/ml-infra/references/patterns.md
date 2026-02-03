# Implementation Patterns

## Table of Contents
1. [Component Registry](#registry)
2. [Trainer Base Class](#trainer)
3. [Distributed Training](#distributed)
4. [Checkpoint System](#checkpoint)
5. [Logging & Metrics](#logging)
6. [Data Pipeline](#data)

---

## Component Registry {#registry}

### Basic Registry

```python
# src/core/registry.py
from typing import TypeVar, Generic, Callable, Any

T = TypeVar('T')

class Registry(Generic[T]):
    """Generic registry for config-driven instantiation."""
    
    def __init__(self, name: str):
        self._name = name
        self._modules: dict[str, type[T]] = {}
    
    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a class."""
        def decorator(cls: type[T]) -> type[T]:
            if name in self._modules:
                raise ValueError(f"{name} already registered in {self._name}")
            self._modules[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> type[T]:
        """Get registered class by name."""
        if name not in self._modules:
            raise KeyError(f"{name} not found in {self._name}. "
                          f"Available: {list(self._modules.keys())}")
        return self._modules[name]
    
    def build(self, config) -> T:
        """Instantiate from config with 'type' field."""
        cls = self.get(config.type)
        # Pass config without 'type' field
        params = {k: v for k, v in config.items() if k != 'type'}
        return cls(**params)
    
    def list(self) -> list[str]:
        """List all registered names."""
        return list(self._modules.keys())


# Global registries
MODELS = Registry[nn.Module]("models")
ALGORITHMS = Registry("algorithms")
DATASETS = Registry("datasets")
REWARDS = Registry("rewards")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
```

### Usage Pattern

```python
# src/algorithms/ppo/trainer.py
from src.core.registry import ALGORITHMS

@ALGORITHMS.register("ppo")
class PPOTrainer(BaseTrainer):
    def __init__(
        self,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef


# Instantiation in main
config = load_config('configs/exp001.yaml')
trainer = ALGORITHMS.build(config.algorithm)
```

---

## Trainer Base Class {#trainer}

```python
# src/algorithms/base.py
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

@dataclass
class TrainOutput:
    loss: torch.Tensor
    metrics: dict[str, float]

class BaseTrainer(ABC):
    """Abstract base for all training algorithms."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None,
        config: Any,
        logger: Any,
        exp_dir: Path,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.logger = logger
        self.exp_dir = exp_dir
        
        self.global_step = 0
        self.epoch = 0
    
    # === Abstract methods (MUST implement) ===
    
    @abstractmethod
    def compute_loss(self, batch: dict) -> TrainOutput:
        """
        Compute loss and metrics for a single batch.
        
        Returns:
            TrainOutput with loss tensor and metrics dict
        """
        pass
    
    # === Hooks (CAN override) ===
    
    def on_train_start(self):
        """Called before training loop."""
        pass
    
    def on_train_end(self):
        """Called after training loop."""
        pass
    
    def on_step_start(self, step: int, batch: dict):
        """Called before each training step."""
        pass
    
    def on_step_end(self, step: int, output: TrainOutput):
        """Called after each training step."""
        pass
    
    def on_eval_start(self):
        """Called before evaluation."""
        pass
    
    def on_eval_end(self, metrics: dict):
        """Called after evaluation."""
        pass
    
    # === Core training logic (RARELY override) ===
    
    def training_step(self, batch: dict) -> TrainOutput:
        """Single training step with gradient accumulation."""
        self.model.train()
        
        # Forward pass
        output = self.compute_loss(batch)
        loss = output.loss / self.config.training.gradient_accumulation
        
        # Backward pass
        loss.backward()
        
        return output
    
    def optimizer_step(self):
        """Optimizer step with gradient clipping."""
        if self.config.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    def train(self):
        """Main training loop."""
        self.on_train_start()
        
        accum_steps = self.config.training.gradient_accumulation
        total_steps = self.config.training.total_steps
        
        self.optimizer.zero_grad()
        data_iter = iter(self.train_dataloader)
        
        while self.global_step < total_steps:
            # Accumulate gradients
            for accum_idx in range(accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.epoch += 1
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)
                
                self.on_step_start(self.global_step, batch)
                output = self.training_step(batch)
            
            # Optimizer step after accumulation
            self.optimizer_step()
            self.global_step += 1
            
            self.on_step_end(self.global_step, output)
            
            # Logging
            if self.global_step % self.config.training.log_interval == 0:
                self.log_metrics(output.metrics)
            
            # Evaluation
            if (self.eval_dataloader and 
                self.global_step % self.config.training.eval_interval == 0):
                eval_metrics = self.evaluate()
                self.on_eval_end(eval_metrics)
            
            # Checkpointing
            if self.global_step % self.config.training.save_interval == 0:
                self.save_checkpoint()
        
        self.on_train_end()
    
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation loop."""
        self.model.eval()
        self.on_eval_start()
        
        all_metrics = []
        for batch in self.eval_dataloader:
            output = self.compute_loss(batch)
            all_metrics.append(output.metrics)
        
        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[f"eval/{key}"] = sum(values) / len(values)
        
        self.model.train()
        return aggregated
    
    def log_metrics(self, metrics: dict):
        """Log metrics to logger."""
        self.logger.log({
            **metrics,
            'step': self.global_step,
            'epoch': self.epoch,
            'lr': self.scheduler.get_last_lr()[0],
        })
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        ckpt_dir = self.exp_dir / 'checkpoints' / f'step_{self.global_step}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
        }, ckpt_dir / 'checkpoint.pt')
    
    def load_checkpoint(self, ckpt_path: Path):
        """Load training checkpoint."""
        ckpt = torch.load(ckpt_path / 'checkpoint.pt')
        
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.global_step = ckpt['global_step']
        self.epoch = ckpt['epoch']
```

---

## Distributed Training {#distributed}

```python
# src/distributed/strategy.py
from abc import ABC, abstractmethod
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class DistributedStrategy(ABC):
    """Abstract distributed training strategy."""
    
    @abstractmethod
    def setup(self):
        """Initialize distributed environment."""
        pass
    
    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, model: nn.Module, path: Path):
        """Save distributed checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, model: nn.Module, path: Path):
        """Load distributed checkpoint."""
        pass
    
    @property
    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0
    
    @property
    def world_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1
    
    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


class FSDPStrategy(DistributedStrategy):
    """Fully Sharded Data Parallel strategy."""
    
    def __init__(self, config):
        self.config = config
    
    def setup(self):
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.rank)
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        return FSDP(
            model,
            sharding_strategy=self.config.sharding_strategy,
            cpu_offload=self.config.cpu_offload,
            mixed_precision=self.config.mixed_precision,
        )
    
    def save_checkpoint(self, model: nn.Module, path: Path):
        # FSDP-specific checkpoint saving
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state_dict = model.state_dict()
            if self.is_main_process:
                torch.save(state_dict, path)
        dist.barrier()


# src/distributed/utils.py
def is_distributed() -> bool:
    return dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def barrier():
    if is_distributed():
        dist.barrier()

def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM):
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor

def all_gather(tensor: torch.Tensor) -> list[torch.Tensor]:
    if not is_distributed():
        return [tensor]
    
    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    return gathered
```

---

## Checkpoint System {#checkpoint}

```python
# src/core/checkpoint.py
from pathlib import Path
from typing import Any
import torch
import json
from dataclasses import dataclass

@dataclass
class CheckpointManager:
    """Manage training checkpoints with rotation."""
    
    exp_dir: Path
    max_checkpoints: int = 5
    
    def __post_init__(self):
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        extra: dict | None = None,
    ):
        """Save checkpoint and rotate old ones."""
        step_dir = self.ckpt_dir / f'step_{step}'
        step_dir.mkdir(exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), step_dir / 'model.pt')
        
        # Save optimizer/scheduler
        torch.save({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': step,
            **(extra or {}),
        }, step_dir / 'training_state.pt')
        
        # Update latest symlink
        latest_link = self.ckpt_dir / 'latest'
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(step_dir.name)
        
        # Rotate checkpoints
        self._rotate_checkpoints()
    
    def load_latest(self) -> Path | None:
        """Get path to latest checkpoint."""
        latest_link = self.ckpt_dir / 'latest'
        if latest_link.exists():
            return self.ckpt_dir / latest_link.resolve().name
        return None
    
    def load(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> dict:
        """Load checkpoint."""
        # Load model
        model.load_state_dict(torch.load(path / 'model.pt'))
        
        # Load training state
        state = torch.load(path / 'training_state.pt')
        
        if optimizer:
            optimizer.load_state_dict(state['optimizer'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler'])
        
        return state
    
    def _rotate_checkpoints(self):
        """Remove old checkpoints keeping max_checkpoints."""
        ckpts = sorted(
            [d for d in self.ckpt_dir.iterdir() if d.is_dir() and d.name.startswith('step_')],
            key=lambda d: int(d.name.split('_')[1])
        )
        
        while len(ckpts) > self.max_checkpoints:
            oldest = ckpts.pop(0)
            shutil.rmtree(oldest)
```

---

## Logging & Metrics {#logging}

```python
# src/core/logging.py
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

class Logger(ABC):
    """Abstract logger interface."""
    
    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None):
        pass
    
    @abstractmethod
    def log_config(self, config: dict):
        pass
    
    @abstractmethod
    def finish(self):
        pass


class WandbLogger(Logger):
    """Weights & Biases logger."""
    
    def __init__(self, project: str, name: str, config: dict, **kwargs):
        import wandb
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            **kwargs
        )
    
    def log(self, metrics: dict[str, Any], step: int | None = None):
        self.run.log(metrics, step=step)
    
    def log_config(self, config: dict):
        self.run.config.update(config)
    
    def finish(self):
        self.run.finish()


class CompositeLogger(Logger):
    """Combine multiple loggers."""
    
    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers
    
    def log(self, metrics: dict[str, Any], step: int | None = None):
        for logger in self.loggers:
            logger.log(metrics, step)
    
    def log_config(self, config: dict):
        for logger in self.loggers:
            logger.log_config(config)
    
    def finish(self):
        for logger in self.loggers:
            logger.finish()


# Metrics aggregation
class MetricsAccumulator:
    """Accumulate and average metrics over steps."""
    
    def __init__(self):
        self._metrics: dict[str, list[float]] = {}
    
    def add(self, metrics: dict[str, float]):
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append(value)
    
    def get_average(self, reset: bool = True) -> dict[str, float]:
        result = {}
        for key, values in self._metrics.items():
            result[key] = sum(values) / len(values)
        
        if reset:
            self._metrics = {}
        
        return result
```

---

## Data Pipeline {#data}

```python
# src/data/datasets/base.py
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Any

class BaseDataset(Dataset, ABC):
    """Base dataset with common functionality."""
    
    def __init__(self, data_path: str | Path, tokenizer: Any, max_length: int):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()
    
    @abstractmethod
    def _load_data(self) -> list[dict]:
        """Load and preprocess data."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Return single item."""
        pass
    
    def __len__(self) -> int:
        return len(self.data)


# src/data/datasets/prompt_dataset.py
@DATASETS.register("prompt")
class PromptDataset(BaseDataset):
    """Dataset of prompts for generation."""
    
    def _load_data(self) -> list[dict]:
        with open(self.data_path) as f:
            return [json.loads(line) for line in f]
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        encoded = self.tokenizer(
            item['prompt'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'prompt_text': item['prompt'],
        }


# src/data/collators.py
from dataclasses import dataclass
import torch

@dataclass
class DataCollator:
    """Collate batch with padding."""
    
    tokenizer: Any
    padding: str = 'longest'
    max_length: int | None = None
    
    def __call__(self, batch: list[dict]) -> dict:
        # Separate tensor and non-tensor fields
        tensor_keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
        other_keys = [k for k in batch[0] if k not in tensor_keys]
        
        result = {}
        
        # Pad tensors
        for key in tensor_keys:
            tensors = [item[key] for item in batch]
            result[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        
        # Collect non-tensors
        for key in other_keys:
            result[key] = [item[key] for item in batch]
        
        return result


# src/data/samplers.py
from torch.utils.data import Sampler, DistributedSampler

class InfiniteSampler(Sampler):
    """Infinite sampler for continuous training."""
    
    def __init__(self, dataset, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
    
    def __iter__(self):
        while True:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
            
            for idx in indices:
                yield idx
            
            self.epoch += 1
    
    def __len__(self):
        return len(self.dataset)
```
