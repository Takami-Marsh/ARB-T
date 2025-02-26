import torch
import asyncio
from arb_t.model import AgenticLLM

async def main():
    # Initialize model with enhanced configuration
    INPUT_SIZE = 768  # Typical embedding size
    VOCAB_SIZE = 32000  # Example vocabulary size
    HIDDEN_SIZE = 512
    
    model = AgenticLLM(
        input_size=INPUT_SIZE,
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        learning_rate=0.001,
        max_sequence_length=1024
    )
    
    print("Demonstrating internet-based learning...")
    print("\nModel configuration:")
    print(f"Input size: {INPUT_SIZE}")
    print(f"Hidden size: {HIDDEN_SIZE}")
    print(f"Number of attention heads: {model.self_attention_layers[0].num_heads}")
    print(f"Number of layers: {len(model.self_attention_layers)}")
    
    # Example 1: Learning about a new topic
    query = "neural attention transformer pytorch"  # Simplified query
    results = await model.search_and_learn(query)
    print(f"\nFetched {len(results)} results about attention mechanisms")
    if results:
        print("\nTop result details:")
        print(f"Title: {results[0].title}")
        print(f"URL: {results[0].url}")
        print(f"Relevance score: {results[0].relevance_score}")
    
    if results:
        print("\nTop result:")
        print(f"Title: {results[0].title}")
        print(f"URL: {results[0].url}")
    
    # Simulate sequence input data
    batch_size = 4
    sequence_length = 32
    embedding_size = HIDDEN_SIZE  # Use hidden size for embeddings
    x = torch.randn(batch_size, sequence_length, embedding_size)  # Input features
    context = torch.randn(batch_size, sequence_length, embedding_size)  # Context features
    
    print("\nProcessing with attention and all cognitive functions...")
    output = model(x, context=context)
    model.update_learning_progress()
    
    print("\nDemonstrating specialized processing...")
    # Process using memory and logic with self-attention
    output_specialized = model(x, active_functions=["memory", "logic"])
    model.update_learning_progress()
    
    # Demonstrate adaptive learning
    print("\nSimulating learning process with internet search...")
    for i in range(5):
        query = f"neural network implementation part {i+1} python"
        results = await model.search_and_learn(query)
        print(f"\nIteration {i+1}: Found {len(results)} relevant resources")
        
        # Process new input with learned knowledge
        x = torch.randn(batch_size, sequence_length, embedding_size)
        output = model(x, context=torch.randn(batch_size, sequence_length, embedding_size))
        model.update_learning_progress()
        
        # Print stats about active learning tasks
        active_tasks = model.learning_controller.active_tasks
        if active_tasks:
            print(f"\nIteration {i+1}")
            print(f"Active learning tasks: {len(active_tasks)}")
            for task in active_tasks:
                print(f"  - {task.compartment}: target performance = {task.target_performance:.3f}")
                
            # Print neuron counts for each compartment
            for name, compartment in model.network.compartments.items():
                print(f"  {name} neurons: {compartment.current_neurons}")
        
    print("\nDemonstrating memory function...")
    # Create multiple patterns to demonstrate memory
    patterns = [torch.randn(1, sequence_length, embedding_size) for _ in range(3)]
    
    print("\nTraining memory with repeated patterns...")
    # Expose each pattern multiple times
    for i, pattern in enumerate(patterns):
        repeated_pattern = pattern.repeat(batch_size, 1, 1)
        print(f"\nPattern {i+1}:")
        
        # Multiple exposures to same pattern with performance tracking
        memory_function = model.functions["memory"]
        start_metrics = len(memory_function.performance_metrics)
        for exposure in range(5):
            output = model(repeated_pattern, active_functions=["memory"])
            current_metrics = memory_function.performance_metrics[start_metrics:]
            recall_rate = sum(current_metrics) / len(current_metrics)
            print(f"  Exposure {exposure + 1} complete (Recall rate: {recall_rate:.2%})")
            
        # Add some variation to test pattern recognition
        noisy_pattern = repeated_pattern + torch.randn_like(repeated_pattern) * 0.1
        print("  Testing with noisy version...")
        output_noisy = model(noisy_pattern, active_functions=["memory"])
    
    memory_function = model.functions["memory"]
    # Print memory statistics
    print("\nMemory Function Analysis:")
    print(f"Total patterns stored: {len(memory_function.memory_buffer)}")
    print(f"Overall recall rate: {sum(memory_function.performance_metrics) / len(memory_function.performance_metrics):.2%}")
    
    # Show pattern recognition stats
    recalled_patterns = {k: v for k, v in memory_function.access_counts.items() if v > 1}
    print(f"\nRecalled patterns: {len(recalled_patterns)}")
    if recalled_patterns:
        print("\nTop recalled patterns:")
        sorted_patterns = sorted(recalled_patterns.items(), key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_patterns[:5]:
            recognition_rate = count / (count + 1)  # Adjust for initial storage
            print(f"  Pattern: {pattern}")
            print(f"  Access count: {count}")
            print(f"  Recognition rate: {recognition_rate:.2%}")
    
    print("\nFinal compartment sizes:")
    for name, compartment in model.network.compartments.items():
        print(f"{name}: {compartment.current_neurons} neurons")

if __name__ == "__main__":
    asyncio.run(main())
