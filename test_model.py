#!/usr/bin/env python3
"""
Test script for the fine-tuned model
Tests the trained model's ability to generate responses in your conversation style
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class ModelTester:
    """Test the fine-tuned model"""
    
    def __init__(self, model_path, base_model="Qwen/Qwen-1_8B"):
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.device = self._detect_device()
        
        print(f"Using device: {self.device}")
        print(f"Loading model from: {self.model_path}")
        
        self.tokenizer, self.model = self.load_model()
        
    def _detect_device(self):
        """Detect the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                test_tensor = torch.zeros(1).to("mps")
                del test_tensor
                return "mps"
            except Exception as e:
                print(f"MPS available but not functional: {e}")
                return "cpu"
        else:
            return "cpu"
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, 
            trust_remote_code=True, 
            padding_side="right"
        )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, self.model_path)
        
        # Move to device
        if self.device != "cpu":
            model = model.to(self.device)
        
        # Set to evaluation mode
        model.eval()
        
        return tokenizer, model
    
    def generate_response(self, prompt, max_length=200, temperature=0.7, do_sample=True):
        """Generate a response to the given prompt"""
        
        # Format the prompt like training data
        formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response = response[len(formatted_prompt):].strip()
        
        return response
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n" + "="*50)
        print("🤖 Interactive Model Testing")
        print("Type 'quit' to exit")
        print("="*50 + "\n")
        
        while True:
            try:
                prompt = input("You: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not prompt:
                    continue
                
                print("🤖 Generating response...")
                response = self.generate_response(prompt)
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def batch_test(self, test_prompts):
        """Test with a batch of predefined prompts"""
        print("\n" + "="*50)
        print("🧪 Batch Testing")
        print("="*50 + "\n")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"Test {i}: {prompt}")
            try:
                response = self.generate_response(prompt)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")
            print("-" * 30)


def run_predefined_tests(tester):
    """Run some predefined tests to evaluate the model"""
    
    test_prompts = [
        "Hello, how are you?",
        "What did you do today?",
        "What's your favorite movie?",
        "Tell me about your weekend",
        "What are you thinking about?",
        "Any plans for tonight?",
        "How was work?",
        "What's for dinner?",
        "Are you free this weekend?",
        "Did you see that movie?"
    ]
    
    tester.batch_test(test_prompts)


def main():
    parser = argparse.ArgumentParser(description="Test the fine-tuned model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./fine_tuned_model/checkpoints/checkpoint-500",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen-1_8B",
        help="Base model used for fine-tuning"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--batch_test",
        action="store_true",
        help="Run predefined batch tests"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (0.1-1.0)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Maximum length of generated responses"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        return
    
    # Initialize tester
    try:
        tester = ModelTester(args.model_path, args.base_model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run tests based on arguments
    if args.batch_test:
        run_predefined_tests(tester)
    
    if args.interactive:
        tester.interactive_test()
    
    # If no specific mode chosen, run both
    if not args.interactive and not args.batch_test:
        print("Running both batch tests and interactive mode...")
        run_predefined_tests(tester)
        tester.interactive_test()


if __name__ == "__main__":
    main()