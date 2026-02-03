# Abstractions and Interfaces

## Core Principle

**Stable interfaces, pluggable implementations.**

Abstract base classes define contracts that:
- All implementations must follow
- Enable swapping algorithms without changing infrastructure
- Provide shared functionality (logging, checkpointing, etc.)

---

## BaseTrainer

The core abstraction for all training algorithms.

```python
# src/algorithms/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Any
import torch
from torch.utils.data import DataLoader

from src.core import Config, Logger
from src.distributed import DistributedStrategy


@dataclass
class Batch:
    """Standard batch format."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None = None
    # Algorithm-specific fields added by subclasses
    

@dataclass
class LossOutput:
    """Standard loss output."""
    loss: torch.Tensor
    metrics: dict[str, float]  # For logging


@dataclass
class StepOutput:
    """Training step output."""
    loss: float
    metrics: dict[str, float]
    lr: float


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(config)
        self.distributed = DistributedStrategy.from_config(config)
        
        # Setup
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.train_dataloader = self._build_dataloader()
        
        # State
        self.global_step = 0
        self.epoch = 0
    
    # ==================== Abstract Methods ====================
    # Subclasses MUST implement these
    
    @abstractmethod
    def compute_loss(self, batch: Batch) -> LossOutput:
        """Compute loss for a batch. Algorithm-specific."""
        pass
    
    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        """Build and return the model(s). Algorithm-specific."""
        pass
    
    @abstractmethod
    def _build_dataloader(self) -> DataLoader:
        """Build training dataloader. Algorithm-specific."""
        pass
    
    # ==================== Optional Overrides ====================
    # Subclasses MAY override these
    
    def training_step(self, batch: Batch) -> StepOutput:
        """Single training step. Override for custom behavior."""
        self.model.train()
        
        # Forward
        loss_output = self.compute_loss(batch)
        loss = loss_output.loss / self.config.training.gradient_accumulation
        
        # Backward
        self.distributed.backward(loss)
        
        # Step (if accumulation complete)
        if (self.global_step + 1) % self.config.training.gradient_accumulation == 0:
            self.distributed.clip_grad_norm(self.model, self.config.training.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return StepOutput(
            loss=loss_output.loss.item(),
            metrics=loss_output.metrics,
            lr=self.scheduler.get_last_lr()[0]
        )
    
    def evaluate(self) -> dict[str, float]:
        """Run evaluation. Override for custom metrics."""
        self.model.eval()
        # Default: return empty, subclass implements
        return {}
    
    def on_step_end(self, step: int, output: StepOutput):
        """Hook called after each step. Override for custom behavior."""
        pass
    
    def on_epoch_end(self, epoch: int):
        """Hook called after each epoch. Override for custom behavior."""
        pass
    
    # ==================== Shared Implementation ====================
    # Usually no need to override
    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.training.total_steps} steps")
        
        while self.global_step < self.config.training.total_steps:
            for batch in self.train_dataloader:
                # Training step
                batch = self._to_device(batch)
                output = self.training_step(batch)
                
                # Logging
                if self.global_step % self.config.logging.log_interval == 0:
                    self.logger.log_metrics(output.metrics, step=self.global_step)
                
                # Evaluation
                if self.global_step % self.config.logging.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    self.logger.log_metrics(eval_metrics, step=self.global_step, prefix="eval")
                
                # Checkpointing
                if self.global_step % self.config.logging.save_interval == 0:
                    self.save_checkpoint()
                
                # Custom hook
                self.on_step_end(self.global_step, output)
                
                self.global_step += 1
                if self.global_step >= self.config.training.total_steps:
                    break
            
            self.epoch += 1
            self.on_epoch_end(self.epoch)
        
        # Final save
        self.save_checkpoint(final=True)
        self.logger.info("Training complete")
    
    def save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        tag = "final" if final else f"step_{self.global_step}"
        path = self.config.experiment.output_dir / "checkpoints" / tag
        
        self.distributed.save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=self.epoch,
        )
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        state = self.distributed.load_checkpoint(path)
        self.global_step = state["step"]
        self.epoch = state["epoch"]
        self.logger.info(f"Resumed from {path} at step {self.global_step}")
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            betas=tuple(self.config.optimizer.betas),
            weight_decay=self.config.optimizer.weight_decay,
        )
    
    def _build_scheduler(self):
        """Build LR scheduler from config."""
        from transformers import get_scheduler
        return get_scheduler(
            self.config.scheduler.name,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=self.config.training.total_steps,
        )
    
    def _to_device(self, batch: Batch) -> Batch:
        """Move batch to device."""
        return self.distributed.to_device(batch)
```

---

## Algorithm-Specific Implementations

### DPO Trainer

```python
# src/algorithms/dpo/trainer.py
from src.algorithms.base import BaseTrainer, Batch, LossOutput
from src.core.registry import ALGORITHMS
from .loss import dpo_loss

@ALGORITHMS.register("dpo")
class DPOTrainer(BaseTrainer):
    """Direct Preference Optimization trainer."""
    
    def __init__(self, config: Config):
        self.beta = config.dpo.beta
        self.label_smoothing = config.dpo.label_smoothing
        self.reference_model = None  # Loaded in _build_model
        super().__init__(config)
    
    def _build_model(self):
        from transformers import AutoModelForCausalLM
        
        # Policy model (trainable)
        policy = AutoModelForCausalLM.from_pretrained(
            self.config.model.path,
            torch_dtype=torch.bfloat16,
        )
        
        # Reference model (frozen)
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.path,
            torch_dtype=torch.bfloat16,
        )
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        return policy
    
    def _build_dataloader(self):
        from src.data.datasets import PreferenceDataset
        dataset = PreferenceDataset(self.config.data)
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=dataset.collate,
        )
    
    def compute_loss(self, batch: Batch) -> LossOutput:
        # Get policy log probs
        policy_logps = self._get_logps(self.model, batch)
        
        # Get reference log probs (no grad)
        with torch.no_grad():
            ref_logps = self._get_logps(self.reference_model, batch)
        
        # Compute DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logps=policy_logps.chosen,
            policy_rejected_logps=policy_logps.rejected,
            reference_chosen_logps=ref_logps.chosen,
            reference_rejected_logps=ref_logps.rejected,
            beta=self.beta,
            label_smoothing=self.label_smoothing,
        )
        
        return LossOutput(loss=loss, metrics=metrics)
    
    def _get_logps(self, model, batch) -> LogProbOutput:
        """Get log probabilities for chosen and rejected."""
        # Implementation details...
        pass
```

### PPO Trainer

```python
# src/algorithms/ppo/trainer.py
from src.algorithms.base import BaseTrainer, Batch, LossOutput, StepOutput
from src.core.registry import ALGORITHMS
from .buffer import ExperienceBuffer
from .loss import ppo_loss

@ALGORITHMS.register("ppo")
class PPOTrainer(BaseTrainer):
    """Proximal Policy Optimization trainer."""
    
    def __init__(self, config: Config):
        self.clip_ratio = config.ppo.clip_ratio
        self.value_coef = config.ppo.value_coef
        self.kl_coef = config.ppo.kl_coef
        self.buffer = ExperienceBuffer(config.ppo.buffer_size)
        super().__init__(config)
    
    def _build_model(self):
        from src.models import PolicyWithValueHead
        return PolicyWithValueHead.from_pretrained(self.config.model.path)
    
    def training_step(self, batch: Batch) -> StepOutput:
        """PPO has a different training loop."""
        # 1. Generate rollouts
        experiences = self.generate_rollouts(batch)
        self.buffer.add(experiences)
        
        # 2. Compute advantages
        advantages = self.compute_advantages(experiences)
        
        # 3. PPO update epochs
        for _ in range(self.config.ppo.update_epochs):
            for minibatch in self.buffer.sample_minibatches():
                loss_output = self.compute_loss(minibatch, advantages)
                # ... optimization step
        
        return StepOutput(...)
    
    def compute_loss(self, batch: Batch) -> LossOutput:
        loss, metrics = ppo_loss(
            # ... PPO-specific arguments
        )
        return LossOutput(loss=loss, metrics=metrics)
```

---

## Data Abstractions

```python
# src/data/datasets/base.py
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    """Base class for all datasets."""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = self._load_tokenizer()
        self.data = self._load_data()
    
    @abstractmethod
    def _load_data(self) -> list:
        """Load and preprocess data."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Get single item."""
        pass
    
    @abstractmethod
    def collate(self, batch: list[dict]) -> Batch:
        """Collate items into batch."""
        pass
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.config.model.path)
```

---

## Model Abstractions

```python
# src/models/base.py
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """Base class for all models."""
    
    @abstractmethod
    def forward(self, **kwargs):
        pass
    
    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str, **kwargs) -> "BaseModel":
        pass
    
    def save_pretrained(self, path: str):
        """Save model weights."""
        pass
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

## When to Use Abstractions

| Scenario | Use Abstract Base Class |
|----------|------------------------|
| Multiple implementations expected | ✅ Yes |
| Shared infrastructure (logging, checkpointing) | ✅ Yes |
| One-off experiment | ❌ No, keep simple |
| External library handles abstraction | ❌ No, wrap if needed |

**Rule: Abstract only what varies, keep shared code in base class.**
