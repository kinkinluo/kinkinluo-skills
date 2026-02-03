# Component Registry

## Core Principle

**Instantiate components from config strings.**

Registry pattern enables:
- Config-driven component selection
- Easy addition of new implementations
- Clean separation of registration and usage

---

## Registry Implementation

```python
# src/core/registry.py
from typing import TypeVar, Generic, Callable, Any

T = TypeVar("T")

class Registry(Generic[T]):
    """Generic component registry."""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, type[T]] = {}
    
    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a class."""
        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise ValueError(f"{name} already registered in {self.name}")
            self._registry[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> type[T]:
        """Get registered class by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"Unknown {self.name}: {name}. Available: {available}")
        return self._registry[name]
    
    def build(self, name: str, **kwargs) -> T:
        """Instantiate registered class."""
        cls = self.get(name)
        return cls(**kwargs)
    
    def list(self) -> list[str]:
        """List all registered names."""
        return list(self._registry.keys())


# Create registries for each component type
ALGORITHMS = Registry[BaseTrainer]("algorithms")
MODELS = Registry("models")
DATASETS = Registry("datasets")
LOSSES = Registry("losses")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
```

---

## Registration Pattern

```python
# src/algorithms/dpo/trainer.py
from src.core.registry import ALGORITHMS
from src.algorithms.base import BaseTrainer

@ALGORITHMS.register("dpo")
class DPOTrainer(BaseTrainer):
    """Direct Preference Optimization trainer."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.beta = config.dpo.beta
        self.loss_type = config.dpo.loss_type
    
    def compute_loss(self, batch: Batch) -> LossOutput:
        # DPO-specific implementation
        ...
```

```python
# src/algorithms/ppo/trainer.py
from src.core.registry import ALGORITHMS
from src.algorithms.base import BaseTrainer

@ALGORITHMS.register("ppo")
class PPOTrainer(BaseTrainer):
    """Proximal Policy Optimization trainer."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.clip_ratio = config.ppo.clip_ratio
        self.value_coef = config.ppo.value_coef
    
    def compute_loss(self, batch: Batch) -> LossOutput:
        # PPO-specific implementation
        ...
```

---

## Auto-Discovery Pattern

```python
# src/algorithms/__init__.py
from src.core.registry import ALGORITHMS

# Import all modules to trigger registration
from . import sft
from . import dpo
from . import ppo
from . import grpo

# Or use auto-discovery
def auto_register():
    """Auto-import all algorithm modules."""
    import importlib
    from pathlib import Path
    
    package_dir = Path(__file__).parent
    for path in package_dir.iterdir():
        if path.is_dir() and not path.name.startswith("_"):
            importlib.import_module(f".{path.name}", package=__name__)

auto_register()

__all__ = ["ALGORITHMS"]
```

---

## Usage in Training Script

```python
# scripts/train.py
from src.core import Config
from src.algorithms import ALGORITHMS

def main():
    config = Config.from_cli()
    
    # Build trainer from config
    trainer = ALGORITHMS.build(
        config.algorithm.name,  # e.g., "dpo"
        config=config
    )
    
    trainer.train()
```

```yaml
# configs/experiments/exp001.yaml
algorithm:
  name: dpo  # Selects DPOTrainer

dpo:
  beta: 0.1
```

---

## Nested Registry Pattern

For complex components with sub-components:

```python
# src/models/losses/__init__.py
LOSSES = Registry("losses")

@LOSSES.register("cross_entropy")
class CrossEntropyLoss:
    ...

@LOSSES.register("focal")
class FocalLoss:
    ...

@LOSSES.register("dpo")
class DPOLoss:
    ...
```

```python
# src/algorithms/base.py
from src.models.losses import LOSSES

class BaseTrainer:
    def __init__(self, config: Config):
        # Build loss from config
        self.loss_fn = LOSSES.build(
            config.loss.name,
            **config.loss.to_dict()
        )
```

```yaml
# Config
loss:
  name: dpo
  beta: 0.1
  label_smoothing: 0.0
```

---

## Registry with Config Validation

```python
# src/core/registry.py (enhanced)
from pydantic import BaseModel
from typing import Type

class Registry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, type[T]] = {}
        self._configs: dict[str, Type[BaseModel]] = {}
    
    def register(
        self, 
        name: str, 
        config_cls: Type[BaseModel] | None = None
    ):
        """Register with optional config schema."""
        def decorator(cls: type[T]) -> type[T]:
            self._registry[name] = cls
            if config_cls:
                self._configs[name] = config_cls
            return cls
        return decorator
    
    def build(self, name: str, config: Config | dict = None, **kwargs) -> T:
        """Build with config validation."""
        cls = self.get(name)
        
        # Validate config if schema exists
        if name in self._configs and config:
            config_dict = config.to_dict() if hasattr(config, "to_dict") else config
            validated = self._configs[name](**config_dict)
            return cls(config=validated, **kwargs)
        
        return cls(config=config, **kwargs)
```

```python
# Usage with validation
from pydantic import BaseModel

class DPOConfig(BaseModel):
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"

@ALGORITHMS.register("dpo", config_cls=DPOConfig)
class DPOTrainer(BaseTrainer):
    def __init__(self, config: DPOConfig):
        self.beta = config.beta  # Type-safe access
```

---

## Best Practices

### 1. One Registry Per Component Type
```python
# Good
ALGORITHMS = Registry("algorithms")
MODELS = Registry("models")
LOSSES = Registry("losses")

# Bad - mixing component types
COMPONENTS = Registry("components")  # Too generic
```

### 2. Register at Module Level
```python
# Good - registration happens on import
@ALGORITHMS.register("dpo")
class DPOTrainer:
    ...

# Bad - registration in function
def setup():
    ALGORITHMS.register("dpo")(DPOTrainer)  # Easy to forget
```

### 3. Fail Fast on Missing Registration
```python
def get(self, name: str) -> type[T]:
    if name not in self._registry:
        available = list(self._registry.keys())
        raise KeyError(
            f"Unknown {self.name}: '{name}'. "
            f"Available: {available}. "
            f"Did you forget to import the module?"
        )
    return self._registry[name]
```

### 4. Support Both Class and Instance Registration
```python
class Registry(Generic[T]):
    def register(self, name: str):
        def decorator(cls_or_instance):
            if isinstance(cls_or_instance, type):
                self._registry[name] = cls_or_instance
            else:
                # Register instance factory
                self._registry[name] = lambda **kwargs: cls_or_instance
            return cls_or_instance
        return decorator
```
