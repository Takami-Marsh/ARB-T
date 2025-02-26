# ARB-T: Agentic Learning Language Model

An implementation of a self-learning language model that can initiate its own learning processes and dynamically modify its neural architecture.

## Key Features

- **Self-Initiated Learning**: The model monitors its own performance and initiates learning processes when needed
- **Dynamic Neural Architecture**: Can add neurons to specific compartments based on performance requirements
- **Compartmentalized Processing**: Separate neural compartments for different cognitive functions:
  - Memory processing
  - Language understanding
  - Logical reasoning
- **Selective Training**: Learning processes target specific compartments, allowing focused improvement of individual capabilities

## Architecture

### Core Components

1. **Neural Architecture (`neural_architecture.py`)**
   - DynamicLayer: Base neural layer that can expand by adding neurons
   - CompartmentalizedNetwork: Network structure with separate compartments for different functions

2. **Self-Learning System (`self_learning.py`)**
   - PerformanceMonitor: Tracks performance metrics for each compartment
   - SelfLearningController: Manages learning processes and neural expansion
   - LearningTask: Represents self-initiated learning objectives

3. **Cognitive Functions (`model.py`)**
   - MemoryFunction: Handles information storage and retrieval
   - LanguageFunction: Processes linguistic inputs
   - LogicFunction: Manages logical reasoning tasks

## How It Works

### Self-Learning Process

1. **Performance Monitoring**
   - Each cognitive function tracks its performance metrics
   - The PerformanceMonitor analyzes trends and identifies needs for improvement

2. **Learning Initiation**
   - When performance drops below thresholds, learning tasks are created
   - Learning tasks target specific compartments with customized parameters

3. **Neural Expansion**
   - After repeated performance issues, new neurons are added
   - Expansion is controlled and specific to struggling compartments

### Memory Management

- Memory is compartmentalized to avoid catastrophic forgetting
- Only relevant portions of the network are modified during learning
- Access patterns are tracked to optimize memory usage

## Usage

```python
from arb_t.model import AgenticLLM

# Initialize model
model = AgenticLLM(
    input_size=768,
    vocab_size=32000,
    hidden_size=512
)

# Process input with specific cognitive functions
output = model(input_tensor, active_functions=["memory", "logic"])

# Update learning progress
model.update_learning_progress()
```

See `examples/basic_usage.py` for a complete demonstration of the model's capabilities.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.21+
- transformers 4.30+

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
arb_t/
├── core/
│   ├── neural_architecture.py  # Base neural network components
│   └── self_learning.py       # Self-learning mechanisms
├── model.py                   # Main model implementation
└── examples/
    └── basic_usage.py        # Usage examples
```

## Limitations and Future Work

- Currently simulates some aspects of cognitive functions
- Future work could include:
  - More sophisticated memory management
  - Advanced logical reasoning capabilities
  - Integration with external knowledge bases
  - Improved self-learning strategies
  - Meta-learning capabilities
