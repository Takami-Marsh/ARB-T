import asyncio
import os
import torch
from arb_t.model import AgenticLLM

async def chat_loop(model: AgenticLLM):
    print("\nARB-T Chat Interface")
    print("Type 'exit' to quit, 'save' to save the model, or 'load' to load saved weights.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'save':
                model.save_weights('weights/model.pt')
                print("Model weights saved successfully.")
                continue
            elif user_input.lower() == 'load':
                model.load_weights('weights/model.pt')
                print("Model weights loaded successfully.")
                continue
                
            # Process user input
            results = await model.search_and_learn(user_input)
            
            # Generate response using the model based on user input
            input_ids = model.tokenizer.encode(user_input)
            input_embed = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
            context = torch.randn(1, 32, model.hidden_size)  # Placeholder context
            
            # Process through model
            output = model(input_embed, context=context)
            response = model.tokenizer.decode(output.argmax(dim=-1).squeeze().tolist())
            
            print("\nARB-T: " + response)
            if results:
                print(f"Found {len(results)} relevant sources.")
                print("\nTop result:")
                print(f"Title: {results[0].title}")
                print(f"Source: {results[0].url}")
            
            memory = model.functions["memory"]
            recall_rate = sum(memory.performance_metrics[-5:]) / 5 if memory.performance_metrics else 0
            print(f"\nMemory recall rate: {recall_rate:.2%}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

async def main():
    # Initialize model
    INPUT_SIZE = 768
    VOCAB_SIZE = 32000
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
    
    # Create weights directory if it doesn't exist
    os.makedirs('weights', exist_ok=True)
    
    # Start chat interface
    await chat_loop(model)

if __name__ == "__main__":
    asyncio.run(main())
