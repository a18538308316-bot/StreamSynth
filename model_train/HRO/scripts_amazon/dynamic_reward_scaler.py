#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic reward scaling and tracing utilities for RL training.
Provides:
  - MovingAverageTracker: track moving averages of raw metrics
  - DynamicWeightScheduler: stage-aware / step-aware weight interpolation
  - RewardTracer: append raw component metrics to a JSONL trace file
Usage (inside reward functions):
  from .dynamic_reward_scaler import tracer, weight_scheduler
  tracer.log(step, task="amazon", component="sentiment", raw_score=score, final_reward=final_reward,
             extra={"target_sentiment": target_sentiment})
Configuration is environment-driven or via set_dynamic_reward_config().
"""
import os, json, math, threading, time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

_lock = threading.Lock()

@dataclass
class MovingAverageTracker:
    decay: float = 0.98
    values: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def update(self, key: str, val: float) -> float:
        prev = self.values.get(key, val)
        new_v = self.decay * prev + (1 - self.decay) * val
        self.values[key] = new_v
        self.counts[key] = self.counts.get(key, 0) + 1
        return new_v

    def get(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)

@dataclass
class DynamicWeightScheduler:
    base_weights: Dict[str, float]
    max_scale: Dict[str, float]
    warmup_steps: int = 200
    plateau_shift_steps: int = 2000
    min_floor: float = 0.05

    def get_weight(self, key: str, step: int) -> float:
        base = self.base_weights.get(key, 0.0)
        target_scale = self.max_scale.get(key, base)
        # Warmup interpolation
        if step < self.warmup_steps:
            w = base + (target_scale - base) * (step / max(1, self.warmup_steps))
        else:
            # After warmup, allow slight sinusoidal modulation to avoid stagnation
            phase = (step - self.warmup_steps) / max(1, self.plateau_shift_steps)
            mod = 0.1 * math.sin(2 * math.pi * phase)
            w = target_scale * (1 + mod)
        return max(self.min_floor, w)

class RewardTracer:
    def __init__(self, trace_path: str, enable: bool = True, flush_interval: int = 1):
        self.trace_path = trace_path
        self.enable = enable
        self.flush_interval = flush_interval
        self.buffer = []
        self.last_flush = time.time()
        if self.enable:
            os.makedirs(os.path.dirname(self.trace_path), exist_ok=True)

    def log(self, step: int, task: str, component: str, raw_score: float, final_reward: float, extra: Optional[Dict[str, Any]] = None):
        if not self.enable:
            return
        rec = {
            "step": step,
            "task": task,
            "component": component,
            "raw_score": raw_score,
            "final_reward": final_reward,
            "timestamp": time.time(),
        }
        if extra:
            rec.update(extra)
        self.buffer.append(rec)
        if len(self.buffer) >= self.flush_interval:
            self._flush()

    def _flush(self):
        with _lock:
            if not self.buffer:
                return
            with open(self.trace_path, 'a', encoding='utf-8') as f:
                for item in self.buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.buffer = []
            self.last_flush = time.time()

# Global singletons
_tracer: Optional[RewardTracer] = None
_weight_scheduler: Optional[DynamicWeightScheduler] = None
_ma_tracker = MovingAverageTracker()
_config_applied = False

DEFAULT_BASE_WEIGHTS = {
    "sentiment": 0.25,
    "attribute": 0.15,
    "length": 0.10,
    "generation_quality": 0.30,
    "diversity": 0.30
}

DEFAULT_MAX_SCALE = {
    "sentiment": 0.40,
    "attribute": 0.30,
    "length": 0.20,
    "generation_quality": 0.50,
    "diversity": 0.35
}

def set_dynamic_reward_config(trace_file: str = "./reward_traces/amazon_reward_trace.jsonl",
                              enable_trace: bool = True,
                              warmup_steps: int = 300,
                              plateau_shift_steps: int = 3000,
                              decay: float = 0.98):
    global _tracer, _weight_scheduler, _ma_tracker, _config_applied
    _tracer = RewardTracer(trace_file, enable=enable_trace, flush_interval=10)
    _weight_scheduler = DynamicWeightScheduler(
        base_weights=DEFAULT_BASE_WEIGHTS,
        max_scale=DEFAULT_MAX_SCALE,
        warmup_steps=warmup_steps,
        plateau_shift_steps=plateau_shift_steps,
        min_floor=0.02
    )
    _ma_tracker.decay = decay
    _config_applied = True
    print(f"[DynamicReward] Config applied: trace_file={trace_file}, trace_enabled={enable_trace}")

@property
def tracer():
    return _tracer

@property
def weight_scheduler():
    return _weight_scheduler

def get_scaled_weight(key: str, step: int) -> float:
    if _weight_scheduler is None:
        return DEFAULT_BASE_WEIGHTS.get(key, 0.0)
    return _weight_scheduler.get_weight(key, step)

def update_moving_average(key: str, val: float) -> float:
    return _ma_tracker.update(key, val)

__all__ = [
    'set_dynamic_reward_config', 'tracer', 'weight_scheduler', 'get_scaled_weight', 'update_moving_average'
]
