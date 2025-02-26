import torch
import torch.nn as nn
from typing import Dict, List

class DynamicLayer(nn.Module):
    """A neural network layer that can dynamically add neurons."""
    
    def __init__(self, input_size: int, initial_neurons: int):
        super().__init__()
        self.input_size = input_size
        self.current_neurons = initial_neurons
        self.weight = nn.Parameter(torch.randn(initial_neurons, input_size) / input_size**0.5)
        self.bias = nn.Parameter(torch.zeros(initial_neurons))
        self.output_projection = None
        
    def add_neurons(self, num_neurons: int):
        """Dynamically add new neurons to the layer."""
        new_weights = torch.randn(num_neurons, self.input_size) / self.input_size**0.5
        new_bias = torch.zeros(num_neurons)
        
        self.weight = nn.Parameter(torch.cat([self.weight, new_weights]))
        self.bias = nn.Parameter(torch.cat([self.bias, new_bias]))
        self.current_neurons += num_neurons
        
        # Reset output projection
        self.output_projection = None
        
    def ensure_output_size(self, target_size: int):
        """Ensure output has the correct size using a learnable projection."""
        if self.output_projection is None or self.output_projection.out_features != target_size:
            self.output_projection = nn.Linear(self.current_neurons, target_size)
            if self.weight.is_cuda:
                self.output_projection = self.output_projection.cuda()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.output_projection is not None:
            output = self.output_projection(output)
        return output

class CompartmentalizedNetwork(nn.Module):
    """Neural network with separate compartments for different cognitive functions."""
    
    def __init__(self, input_size: int, compartments: Dict[str, int]):
        """
        Args:
            input_size: Size of input features
            compartments: Dictionary mapping compartment names to their initial neuron counts
        """
        super().__init__()
        self.input_size = input_size
        self.compartments = nn.ModuleDict({
            name: DynamicLayer(input_size, neurons)
            for name, neurons in compartments.items()
        })
        
        # Attention mechanism to weight compartment outputs
        self.attention = nn.Parameter(torch.ones(len(compartments)) / len(compartments))
        
    def expand_compartment(self, compartment_name: str, new_neurons: int):
        """Add neurons to a specific compartment."""
        if compartment_name not in self.compartments:
            raise ValueError(f"Compartment {compartment_name} does not exist")
        self.compartments[compartment_name].add_neurons(new_neurons)
        
    def forward(self, x: torch.Tensor, active_compartments: List[str]) -> torch.Tensor:
        """Forward pass through active compartments."""
        outputs = []
        attention_weights = []
        
        for name in active_compartments:
            # Ensure compartment exists
            if name not in self.compartments:
                raise ValueError(f"Compartment {name} does not exist")
            
            # Ensure output size matches input size for residual connections
            self.compartments[name].ensure_output_size(self.input_size)
            
            # Process through compartment
            output = self.compartments[name](x)
            outputs.append(output)
            
            # Get attention weight
            attention_weights.append(
                self.attention[list(self.compartments.keys()).index(name)]
            )
        
        # Combine outputs using attention weights
        attention_weights = torch.softmax(torch.stack(attention_weights), dim=0)
        weighted_outputs = torch.stack(
            [w * o for w, o in zip(attention_weights, outputs)]
        )
        
        return weighted_outputs.sum(dim=0)
