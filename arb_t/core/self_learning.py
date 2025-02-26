import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .neural_architecture import CompartmentalizedNetwork

@dataclass
class LearningTask:
    """Represents a self-initiated learning task."""
    compartment: str
    importance: float
    target_performance: float
    learning_rate: float
    max_new_neurons: int

class PerformanceMonitor:
    """Monitors and analyzes model performance to trigger learning."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history: Dict[str, List[float]] = {}
        self.improvement_thresholds: Dict[str, float] = {}
        
    def update(self, compartment: str, performance: float):
        """Update performance history for a compartment."""
        if compartment not in self.performance_history:
            self.performance_history[compartment] = []
            self.improvement_thresholds[compartment] = 0.05  # Initial threshold
            
        history = self.performance_history[compartment]
        history.append(performance)
        if len(history) > self.window_size:
            history.pop(0)
            
    def needs_improvement(self, compartment: str) -> Tuple[bool, float]:
        """Determine if a compartment needs improvement."""
        if compartment not in self.performance_history:
            return False, 0.0
            
        history = self.performance_history[compartment]
        if len(history) < self.window_size // 2:
            return False, 0.0
            
        recent_avg = np.mean(history[-self.window_size//4:])
        older_avg = np.mean(history[:-self.window_size//4])
        
        improvement = recent_avg - older_avg
        threshold = self.improvement_thresholds[compartment]
        
        return improvement < threshold, abs(improvement)

class SelfLearningController:
    """Controls self-initiated learning processes."""
    
    def __init__(
        self,
        network: CompartmentalizedNetwork,
        base_learning_rate: float = 0.001,
        min_confidence: float = 0.7
    ):
        self.network = network
        self.base_learning_rate = base_learning_rate
        self.min_confidence = min_confidence
        self.performance_monitor = PerformanceMonitor()
        self.active_tasks: List[LearningTask] = []
        
        # Learning state for each compartment
        self.learning_states: Dict[str, Dict] = {
            name: {
                "consecutive_failures": 0,
                "last_neuron_addition": 0,
                "performance_threshold": 0.8
            }
            for name in network.compartments.keys()
        }
    
    def evaluate_learning_need(self, compartment: str, current_performance: float) -> Optional[LearningTask]:
        """Evaluate if a compartment needs learning and create a task if necessary."""
        self.performance_monitor.update(compartment, current_performance)
        needs_improvement, improvement_gap = self.performance_monitor.needs_improvement(compartment)
        
        if not needs_improvement:
            return None
            
        state = self.learning_states[compartment]
        
        # Determine task parameters based on learning history
        importance = min(1.0, 0.5 + improvement_gap)
        learning_rate = self.base_learning_rate * (1 + state["consecutive_failures"] * 0.2)
        max_new_neurons = int(10 * (1 + improvement_gap))
        
        # Create learning task
        return LearningTask(
            compartment=compartment,
            importance=importance,
            target_performance=current_performance + improvement_gap + 0.1,
            learning_rate=learning_rate,
            max_new_neurons=max_new_neurons
        )
    
    def initiate_learning(self, task: LearningTask):
        """Begin a self-initiated learning process."""
        if task.compartment not in self.network.compartments:
            raise ValueError(f"Unknown compartment: {task.compartment}")
            
        # Add new neurons if performance is consistently poor
        state = self.learning_states[task.compartment]
        if state["consecutive_failures"] >= 3:
            new_neurons = min(
                task.max_new_neurons,
                int(self.network.compartments[task.compartment].current_neurons * 0.1)
            )
            self.network.expand_compartment(task.compartment, new_neurons)
            state["last_neuron_addition"] = 0
            state["consecutive_failures"] = 0
        
        self.active_tasks.append(task)
    
    def update_learning_state(self, compartment: str, success: bool):
        """Update learning state based on task outcome."""
        state = self.learning_states[compartment]
        if success:
            state["consecutive_failures"] = 0
            state["performance_threshold"] *= 1.05  # Increase expectations
        else:
            state["consecutive_failures"] += 1
            state["last_neuron_addition"] += 1
