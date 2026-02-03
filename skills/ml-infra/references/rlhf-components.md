# RLHF / Post-Training Components

## Table of Contents
1. [PPO Trainer](#ppo)
2. [DPO Trainer](#dpo)
3. [GRPO Trainer](#grpo)
4. [Reward Models](#reward)
5. [Reference Model](#reference)
6. [Generation Pipeline](#generation)
7. [Rollout Buffer](#rollout)

---

## PPO Trainer {#ppo}

```python
# src/algorithms/ppo/trainer.py
from src.core.registry import ALGORITHMS
from src.algorithms.base import BaseTrainer, TrainOutput

@ALGORITHMS.register("ppo")
class PPOTrainer(BaseTrainer):
    """Proximal Policy Optimization for RLHF."""
    
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        reward_model: nn.Module,
        value_head: nn.Module,
        # PPO hyperparameters
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.1,
        kl_target: float | None = None,
        num_ppo_epochs: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.policy = policy_model
        self.reference = reference_model
        self.reward_model = reward_model
        self.value_head = value_head
        
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.kl_target = kl_target
        self.num_ppo_epochs = num_ppo_epochs
        
        self.rollout_buffer = RolloutBuffer()
    
    def collect_rollouts(self, prompts: list[str]) -> RolloutBatch:
        """Generate responses and compute rewards."""
        self.policy.eval()
        
        with torch.no_grad():
            # Generate responses
            responses, log_probs, values = self.generate_with_values(prompts)
            
            # Compute rewards
            rewards = self.reward_model.compute_reward(prompts, responses)
            
            # Compute reference log probs for KL penalty
            ref_log_probs = self.reference.compute_log_probs(prompts, responses)
            
            # Compute KL penalty
            kl = log_probs - ref_log_probs
            rewards_with_kl = rewards - self.kl_coef * kl
            
            # Compute advantages (GAE)
            advantages, returns = self.compute_gae(rewards_with_kl, values)
        
        return RolloutBatch(
            prompts=prompts,
            responses=responses,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            kl=kl,
            advantages=advantages,
            returns=returns,
        )
    
    def compute_loss(self, batch: RolloutBatch) -> TrainOutput:
        """PPO clipped objective."""
        # Current policy log probs
        new_log_probs = self.policy.compute_log_probs(
            batch.prompts, batch.responses
        )
        
        # Policy loss with clipping
        ratio = torch.exp(new_log_probs - batch.log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        policy_loss = -torch.min(
            ratio * batch.advantages,
            clipped_ratio * batch.advantages
        ).mean()
        
        # Value loss
        new_values = self.value_head(batch.responses)
        value_loss = F.mse_loss(new_values, batch.returns)
        
        # Entropy bonus
        entropy = self.compute_entropy(new_log_probs)
        
        # Total loss
        loss = (
            policy_loss 
            + self.value_coef * value_loss 
            - self.entropy_coef * entropy
        )
        
        metrics = {
            'loss/policy': policy_loss.item(),
            'loss/value': value_loss.item(),
            'loss/entropy': entropy.item(),
            'ppo/ratio_mean': ratio.mean().item(),
            'ppo/kl_mean': batch.kl.mean().item(),
            'ppo/reward_mean': batch.rewards.mean().item(),
            'ppo/advantage_mean': batch.advantages.mean().item(),
        }
        
        return TrainOutput(loss=loss, metrics=metrics)
    
    def train(self):
        """PPO training loop with rollout collection."""
        for step in range(self.config.training.total_steps):
            # Collect rollouts
            prompts = self.sample_prompts()
            rollouts = self.collect_rollouts(prompts)
            
            # Multiple PPO epochs on same rollouts
            for epoch in range(self.num_ppo_epochs):
                for mini_batch in self.create_minibatches(rollouts):
                    output = self.compute_loss(mini_batch)
                    output.loss.backward()
                    self.optimizer_step()
            
            # Adaptive KL penalty
            if self.kl_target is not None:
                self.update_kl_coef(rollouts.kl.mean())
            
            self.global_step += 1
            self.log_metrics(output.metrics)
    
    def update_kl_coef(self, current_kl: float):
        """Adaptive KL coefficient."""
        if current_kl > self.kl_target * 1.5:
            self.kl_coef *= 1.5
        elif current_kl < self.kl_target / 1.5:
            self.kl_coef /= 1.5
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae = delta + gamma * lam * last_gae
        
        returns = advantages + values
        return advantages, returns
```

---

## DPO Trainer {#dpo}

```python
# src/algorithms/dpo/trainer.py
@ALGORITHMS.register("dpo")
class DPOTrainer(BaseTrainer):
    """Direct Preference Optimization."""
    
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = 'sigmoid',  # sigmoid | hinge | ipo
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.policy = policy_model
        self.reference = reference_model
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
    
    def compute_loss(self, batch: dict) -> TrainOutput:
        """DPO loss: maximize margin between chosen and rejected."""
        
        # Compute log probs for chosen and rejected
        policy_chosen_logps = self.policy.compute_log_probs(
            batch['prompts'], batch['chosen']
        )
        policy_rejected_logps = self.policy.compute_log_probs(
            batch['prompts'], batch['rejected']
        )
        
        with torch.no_grad():
            ref_chosen_logps = self.reference.compute_log_probs(
                batch['prompts'], batch['chosen']
            )
            ref_rejected_logps = self.reference.compute_log_probs(
                batch['prompts'], batch['rejected']
            )
        
        # Compute log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        # DPO loss
        logits = self.beta * (chosen_logratios - rejected_logratios)
        
        if self.loss_type == 'sigmoid':
            loss = -F.logsigmoid(logits).mean()
        elif self.loss_type == 'hinge':
            loss = torch.relu(1 - logits).mean()
        elif self.loss_type == 'ipo':
            loss = (logits - 1 / (2 * self.beta)) ** 2
            loss = loss.mean()
        
        # Label smoothing
        if self.label_smoothing > 0:
            smooth_loss = -F.logsigmoid(-logits).mean()
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss
        
        # Metrics
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        metrics = {
            'loss/dpo': loss.item(),
            'dpo/reward_margin': reward_margin.item(),
            'dpo/accuracy': accuracy.item(),
            'dpo/chosen_reward': chosen_rewards.mean().item(),
            'dpo/rejected_reward': rejected_rewards.mean().item(),
        }
        
        return TrainOutput(loss=loss, metrics=metrics)
```

---

## GRPO Trainer {#grpo}

```python
# src/algorithms/grpo/trainer.py
@ALGORITHMS.register("grpo")
class GRPOTrainer(BaseTrainer):
    """Group Relative Policy Optimization."""
    
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        reward_model: nn.Module,
        num_generations: int = 4,  # G in paper
        beta: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.policy = policy_model
        self.reference = reference_model
        self.reward_model = reward_model
        self.num_generations = num_generations
        self.beta = beta
    
    def compute_loss(self, batch: dict) -> TrainOutput:
        """GRPO: Use relative rewards within group."""
        prompts = batch['prompts']
        batch_size = len(prompts)
        
        # Generate G responses per prompt
        all_responses = []
        all_rewards = []
        
        with torch.no_grad():
            for _ in range(self.num_generations):
                responses = self.policy.generate(prompts)
                rewards = self.reward_model.compute_reward(prompts, responses)
                all_responses.append(responses)
                all_rewards.append(rewards)
        
        # Stack: [G, batch_size]
        all_rewards = torch.stack(all_rewards)
        
        # Compute group-normalized advantages
        # Shape: [G, batch_size]
        group_mean = all_rewards.mean(dim=0, keepdim=True)
        group_std = all_rewards.std(dim=0, keepdim=True) + 1e-8
        advantages = (all_rewards - group_mean) / group_std
        
        # Compute policy loss
        total_loss = 0
        for g in range(self.num_generations):
            policy_logps = self.policy.compute_log_probs(
                prompts, all_responses[g]
            )
            ref_logps = self.reference.compute_log_probs(
                prompts, all_responses[g]
            )
            
            # GRPO objective
            log_ratio = policy_logps - ref_logps
            loss_g = -(advantages[g] * log_ratio - self.beta * log_ratio ** 2)
            total_loss += loss_g.mean()
        
        loss = total_loss / self.num_generations
        
        metrics = {
            'loss/grpo': loss.item(),
            'grpo/reward_mean': all_rewards.mean().item(),
            'grpo/reward_std': all_rewards.std().item(),
            'grpo/advantage_mean': advantages.mean().item(),
        }
        
        return TrainOutput(loss=loss, metrics=metrics)
```

---

## Reward Models {#reward}

```python
# src/rewards/base.py
from abc import ABC, abstractmethod

class BaseRewardModel(ABC):
    """Abstract base for reward models."""
    
    @abstractmethod
    def compute_reward(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> torch.Tensor:
        """Compute reward scores. Returns [batch_size]."""
        pass


# src/rewards/learned.py
@REWARDS.register("learned")
class LearnedRewardModel(BaseRewardModel):
    """Neural reward model trained on preferences."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def compute_reward(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> torch.Tensor:
        texts = [f"{p}\n{r}" for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        ).to(self.model.device)
        
        outputs = self.model(**inputs)
        return outputs.logits.squeeze(-1)


# src/rewards/rule_based.py
@REWARDS.register("rule_based")
class RuleBasedReward(BaseRewardModel):
    """Rule-based rewards for format, length, etc."""
    
    def __init__(
        self,
        length_penalty: float = 0.0,
        target_length: int = 200,
        format_reward: float = 0.0,
    ):
        self.length_penalty = length_penalty
        self.target_length = target_length
        self.format_reward = format_reward
    
    def compute_reward(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> torch.Tensor:
        rewards = []
        
        for response in responses:
            reward = 0.0
            
            # Length penalty
            if self.length_penalty != 0:
                length_diff = abs(len(response.split()) - self.target_length)
                reward -= self.length_penalty * length_diff
            
            # Format checks
            if self.format_reward != 0:
                if response.strip() and response[0].isupper():
                    reward += self.format_reward
                if response.strip().endswith(('.', '!', '?')):
                    reward += self.format_reward
            
            rewards.append(reward)
        
        return torch.tensor(rewards)


# src/rewards/composite.py
@REWARDS.register("composite")
class CompositeReward(BaseRewardModel):
    """Combine multiple reward signals."""
    
    def __init__(self, rewards: list[dict]):
        self.reward_models = []
        self.weights = []
        
        for reward_config in rewards:
            model = REWARDS.build(reward_config)
            weight = reward_config.get('weight', 1.0)
            self.reward_models.append(model)
            self.weights.append(weight)
    
    def compute_reward(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> torch.Tensor:
        total_reward = 0
        
        for model, weight in zip(self.reward_models, self.weights):
            reward = model.compute_reward(prompts, responses)
            total_reward += weight * reward
        
        return total_reward
```

---

## Reference Model {#reference}

```python
# src/models/reference.py
class ReferenceModel:
    """Frozen reference model for KL penalty."""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def compute_log_probs(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> torch.Tensor:
        """Compute log probabilities of responses given prompts."""
        texts = [f"{p}{r}" for p, r in zip(prompts, responses)]
        
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(self.model.device)
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Compute log probs for response tokens only
        log_probs = self._compute_response_log_probs(
            logits, inputs, prompts, responses
        )
        
        return log_probs
```

---

## Generation Pipeline {#generation}

```python
# src/generation/sampler.py
class GenerationSampler:
    """Efficient batch generation with sampling."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
    
    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        return_log_probs: bool = False,
    ) -> tuple[list[str], torch.Tensor | None]:
        """Generate responses for batch of prompts."""
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            output_scores=return_log_probs,
            return_dict_in_generate=True,
        )
        
        # Decode only new tokens
        prompt_lengths = inputs['attention_mask'].sum(dim=1)
        responses = []
        
        for i, seq in enumerate(outputs.sequences):
            response_tokens = seq[prompt_lengths[i]:]
            response = self.tokenizer.decode(
                response_tokens, skip_special_tokens=True
            )
            responses.append(response)
        
        log_probs = None
        if return_log_probs:
            log_probs = self._compute_log_probs(outputs)
        
        return responses, log_probs
```

---

## Rollout Buffer {#rollout}

```python
# src/algorithms/ppo/rollout_buffer.py
from dataclasses import dataclass
import torch

@dataclass
class RolloutBatch:
    """Single batch of rollout data."""
    prompts: list[str]
    responses: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    kl: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Buffer to store and sample rollout data."""
    
    def __init__(self, buffer_size: int = 1024):
        self.buffer_size = buffer_size
        self.rollouts: list[RolloutBatch] = []
    
    def add(self, rollout: RolloutBatch):
        """Add rollout batch to buffer."""
        self.rollouts.append(rollout)
        
        # Trim if over capacity
        total_samples = sum(len(r.prompts) for r in self.rollouts)
        while total_samples > self.buffer_size and len(self.rollouts) > 1:
            self.rollouts.pop(0)
            total_samples = sum(len(r.prompts) for r in self.rollouts)
    
    def sample_minibatches(
        self,
        minibatch_size: int,
        shuffle: bool = True,
    ) -> list[RolloutBatch]:
        """Create minibatches from buffer."""
        # Concatenate all rollouts
        all_data = self._concatenate_rollouts()
        
        indices = list(range(len(all_data['prompts'])))
        if shuffle:
            random.shuffle(indices)
        
        minibatches = []
        for start in range(0, len(indices), minibatch_size):
            batch_indices = indices[start:start + minibatch_size]
            minibatch = self._create_batch(all_data, batch_indices)
            minibatches.append(minibatch)
        
        return minibatches
    
    def clear(self):
        """Clear buffer."""
        self.rollouts = []
```
