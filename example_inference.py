#!/usr/bin/env python3
"""
Example usage of the inference script
"""

from inference import ModelInference
from pathlib import Path


def test_inference():
    """Test the inference functionality with example prompts"""
    
    # Default model path (adjust if needed)
    model_path = "./fine_tuned_model/final_model"
    
    # Check if model exists
    if not Path(model_path).exists():
        print("Model not found. Please train a model first using:")
        print("python finetuning.py --data_path <your_data_path>")
        return
    
    try:
        # Initialize inference
        print("Loading model...")
        inference = ModelInference(model_path)
        
        # Test prompts
        test_prompts = [
            "Hello, how are you?",
            "What did you do today?",
            "Tell me something interesting",
            "What's your favorite movie?",
            "How was your day?"
        ]
        
        print("\n=== Testing Inference ===")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Prompt: {prompt}")
            response = inference.generate_response(prompt, max_length=100, temperature=0.8)
            print(f"   Response: {response}")
        
        print("\n=== Test Complete ===")
        print("The model is working! You can now use:")
        print("python inference.py --prompt 'Your message here'")
        print("or")
        print("python inference.py  # for interactive mode")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure your model was trained successfully.")


if __name__ == "__main__":
    test_inference()