import os
import torch
import torch.nn as nn
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from .core.neural_architecture import CompartmentalizedNetwork
from .core.self_learning import SelfLearningController, LearningTask
from .core.attention import SelfAttention, CrossAttention, PositionalEncoding
from .core.tokenizer import AdaptiveTokenizer
from .core.internet_search import InternetSearchEngine, SearchResult

class CognitiveFunction:
    """Base class for cognitive functions."""
    
    def __init__(self, name: str, input_size: int, initial_neurons: int):
        self.name = name
        self.input_size = input_size
        self.initial_neurons = initial_neurons
        self.performance_metrics: List[float] = []
    
    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def evaluate_performance(self) -> float:
        if not self.performance_metrics:
            return 0.0
        return sum(self.performance_metrics[-10:]) / min(len(self.performance_metrics), 10)

class MemoryFunction(CognitiveFunction):
    """Handles memory-related processing."""
    
    def __init__(self, input_size: int):
        super().__init__("memory", input_size, initial_neurons=128)
        self.memory_buffer: Dict[str, torch.Tensor] = {}
        self.access_counts: Dict[str, int] = {}
        
    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through memory with pattern matching."""
        # Round values for more stable pattern matching
        key = self._generate_key(x)
        matched = False

        # Check for similar patterns
        for stored_key in list(self.memory_buffer.keys()):
            if self._are_patterns_similar(key, stored_key):
                self.access_counts[stored_key] += 1
                matched = True
                key = stored_key  # Use the existing key
                break

        if not matched:
            self.memory_buffer[key] = x
            self.access_counts[key] = 1
            recall_quality = 0.0
        else:
            recall_quality = 1.0
            
        self.performance_metrics.append(recall_quality)
        return x
        
    def _are_patterns_similar(self, key1: str, key2: str) -> bool:
        """Compare two pattern keys for similarity."""
        # Parse the stats from keys
        stats1 = dict(item.split('=') for item in key1.split(';'))
        stats2 = dict(item.split('=') for item in key2.split(';'))
        
        # Compare numerical values with tolerance
        tolerance = 0.1
        for k in ['mean', 'std', 'max', 'min']:
            if k in stats1 and k in stats2:
                val1 = float(stats1[k])
                val2 = float(stats2[k])
                if abs(val1 - val2) > (abs(val1 + val2) / 2) * tolerance:
                    return False
                    
        # Shape must match exactly
        if stats1.get('shape') != stats2.get('shape'):
            return False
            
        return True
        
    def _generate_key(self, x: torch.Tensor) -> str:
        """Generate a more meaningful key based on pattern characteristics."""
        if x.dim() > 2:
            # For sequences, use multiple statistics
            stats = {
                'mean': x.mean().item(),
                'std': x.std().item(),
                'max': x.max().item(),
                'min': x.min().item(),
                'shape': '_'.join(str(d) for d in x.shape)
            }
            # Format values appropriately based on type
            formatted_stats = []
            for k, v in stats.items():
                if k == 'shape':
                    formatted_stats.append(f"{k}={v}")
                else:
                    formatted_stats.append(f"{k}={v:.4f}")
            return ';'.join(formatted_stats)
        else:
            # For non-sequence inputs, use simpler key
            return f"mean={x.mean().item():.4f};std={x.std().item():.4f}"

class LanguageFunction(CognitiveFunction):
    """Handles language processing."""
    
    def __init__(self, input_size: int, vocab_size: int):
        super().__init__("language", input_size, initial_neurons=256)
        self.vocab_size = vocab_size
        self.token_accuracy: List[float] = []
        
    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        # Language processing logic
        x_flat = x.view(-1, x.size(-1))  # Flatten batch and sequence dimensions
        predicted_tokens = torch.argmax(x_flat, dim=-1)
        # Simulate token prediction accuracy
        accuracy = torch.rand(1).item()  # In real implementation, compare with actual targets
        self.token_accuracy.append(accuracy)
        self.performance_metrics.append(accuracy)
        return x

class LogicFunction(CognitiveFunction):
    """Handles logical reasoning."""
    
    def __init__(self, input_size: int):
        super().__init__("logic", input_size, initial_neurons=192)
        self.reasoning_steps: List[Dict[str, Any]] = []
        
    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        # Logic processing simulation
        consistency = torch.sigmoid(x.mean(dim=-1)).mean()  # Average over sequence
        self.reasoning_steps.append({
            "input": x.detach().cpu(),
            "consistency": consistency.item()
        })
        self.performance_metrics.append(consistency.item())
        return x

class AgenticLLM(nn.Module):
    """Enhanced agentic LLM with internet search and self-learning capabilities."""
    
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        max_sequence_length: int = 1024
    ):
        super().__init__()
        
        # Initialize tokenizer and internet search
        self.tokenizer = AdaptiveTokenizer(max_vocab_size=vocab_size)
        self.search_engine = InternetSearchEngine()
        
        # Initialize cognitive functions with attention
        self.functions = {
            "memory": MemoryFunction(hidden_size),
            "language": LanguageFunction(hidden_size, vocab_size),
            "logic": LogicFunction(hidden_size)
        }
        self.hidden_size = hidden_size  # Store for reference
        
        # Add attention layers
        self.self_attention_layers = nn.ModuleList([
            SelfAttention(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.cross_attention = CrossAttention(hidden_size, num_heads, dropout)
        self.positional_encoding = PositionalEncoding(hidden_size, max_sequence_length)
        
        # Create neural network with compartments for each function
        self.network = CompartmentalizedNetwork(
            input_size=hidden_size,  # Use hidden_size instead of input_size
            compartments={
                name: func.initial_neurons
                for name, func in self.functions.items()
            }
        )
        
        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize self-learning controller
        self.learning_controller = SelfLearningController(
            network=self.network,
            base_learning_rate=learning_rate
        )
        
        # Input and output projections
        self.input_projection = nn.Linear(vocab_size, hidden_size)  # Changed from input_size to vocab_size
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
    async def search_and_learn(self, query: str) -> List[SearchResult]:
        """Perform internet search and learn from results."""
        async with self.search_engine as engine:
            # Generate search query
            if self.learning_controller.active_tasks:
                # Use task-specific context for search
                task = self.learning_controller.active_tasks[0]
                context = {"compartment": task.compartment, "performance": task.target_performance}
                search_query = engine.generate_search_query(query, context)
            else:
                search_query = engine.generate_search_query(query)
            
            # Perform search
            results = await engine.search(search_query)
            
            # Process and learn from results
            for result in results:
                content = await engine.fetch_content(result.url)
                if content:
                    # Split content into manageable chunks
                    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                
                    for chunk in chunks[:5]:  # Process first 5 chunks to avoid overwhelming
                        # Update tokenizer vocabulary
                        self.tokenizer.update_vocab(chunk)
                        
                        # Encode content for learning with strict length limit
                        encoded = self.tokenizer.encode(
                            chunk,
                            max_length=512,  # Shorter sequences for efficient processing
                            truncation=True,
                            padding='max_length'
                        )
                        self._learn_from_content(encoded['input_ids'])
                    
            return results
            
    def _learn_from_content(self, encoded_content: torch.Tensor):
        """Learn from encoded content."""
        with torch.no_grad():
            # Convert input to float tensor
            encoded_float = encoded_content.float()
            
            # Create embedding from token IDs (handle out of range tokens)
            token_ids = encoded_float.long().clamp(0, self.tokenizer.max_vocab_size - 1)
            hidden = torch.nn.functional.one_hot(
                token_ids,
                num_classes=self.tokenizer.max_vocab_size
            ).float()
            
            # Process through model (reshape if needed)
            batch_size = hidden.size(0)
            hidden = hidden.view(batch_size, -1, self.tokenizer.max_vocab_size)
            hidden = self.input_projection(hidden)
            hidden = self.positional_encoding(hidden)
            
            # Self-attention layers
            for attention in self.self_attention_layers:
                attended, _ = attention(hidden)
                hidden = self.layer_norm1(hidden + attended)
                
            # Process through cognitive functions
            for func_name, function in self.functions.items():
                processed = function.process_input(hidden)
                hidden = self.layer_norm2(hidden + processed)
                
    def forward(
        self,
        x: torch.Tensor,
        active_functions: Optional[List[str]] = None,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            if context is not None:
                context = context.unsqueeze(1)
        
        # Project input and add positional encoding
        if x.size(-1) != self.tokenizer.max_vocab_size:
            # Regular input needs to be projected directly
            hidden = x
        else:
            # Token input needs to be projected from one-hot encoding
            hidden = self.input_projection(x)
        hidden = self.positional_encoding(hidden)
        
        # Apply self-attention layers
        for attention in self.self_attention_layers:
            attended, _ = attention(hidden)
            hidden = self.layer_norm1(hidden + self.dropout(attended))
            
        # Apply cross-attention if context is provided
        if context is not None:
            context_attended, _ = self.cross_attention(hidden, context)
            hidden = self.layer_norm2(hidden + self.dropout(context_attended))
        
        # Determine which functions to use
        if active_functions is None:
            active_functions = list(self.functions.keys())
            
        # Process through cognitive functions and neural network
        for func_name in active_functions:
            # Process through cognitive function
            hidden = self.functions[func_name].process_input(hidden)
            
            # Check if learning is needed
            performance = self.functions[func_name].evaluate_performance()
            learning_task = self.learning_controller.evaluate_learning_need(
                func_name, performance
            )
            
            if learning_task:
                self.learning_controller.initiate_learning(learning_task)
            
            # Process through neural network compartment with attention
            batch_size, seq_len, _ = hidden.size()
            hidden_flat = hidden.view(batch_size * seq_len, -1)
            output_flat = self.network(hidden_flat, [func_name])
            # Ensure output has correct hidden size
            if output_flat.size(-1) != self.hidden_size:
                output_flat = nn.functional.linear(
                    output_flat,
                    torch.randn(self.hidden_size, output_flat.size(-1), device=output_flat.device)
                )
            hidden = output_flat.view(batch_size, seq_len, -1)
            
        # Project to output space
        return self.output_projection(hidden)
    
    def update_learning_progress(self):
        """Update learning progress for active tasks."""
        for task in self.learning_controller.active_tasks[:]:
            performance = self.functions[task.compartment].evaluate_performance()
            success = performance >= task.target_performance
            
            self.learning_controller.update_learning_state(task.compartment, success)
            
            if success:
                self.learning_controller.active_tasks.remove(task)
                
    def save_weights(self, path: str):
        """Save model weights and states."""
        state_dict = {
            'model': self.state_dict(),
            'network': self.network.state_dict(),
            'functions': {
                name: func.__dict__ 
                for name, func in self.functions.items()
            },
            'learning_controller': self.learning_controller.__dict__
        }
        torch.save(state_dict, path)
        
    def load_weights(self, path: str):
        """Load model weights and states."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved weights found at {path}")
            
        state_dict = torch.load(path)
        
        # Load main model weights
        self.load_state_dict(state_dict['model'])
        
        # Load network weights
        self.network.load_state_dict(state_dict['network'])
        
        # Load function states
        for name, state in state_dict['functions'].items():
            if name in self.functions:
                self.functions[name].__dict__.update(state)
                
        # Load learning controller state
        self.learning_controller.__dict__.update(state_dict['learning_controller'])
