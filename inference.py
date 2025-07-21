#!/usr/bin/env python3
"""
Simple inference script for fine-tuned GPT-2/Qwen model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import json
from pathlib import Path


class ModelInference:
    def __init__(self, model_path, device=None):
        self.model_path = Path(model_path)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    test_tensor = torch.zeros(1).to("mps")
                    del test_tensor
                    self.device = "mps"
                except:
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print(f"Loading model from {self.model_path}")
        
        # Load training config if available
        config_path = self.model_path / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                base_model = config.get("base_model", "gpt2-medium")
                print(f"Base model: {base_model}")
        else:
            base_model = "gpt2-medium"
            print("Using default base model: gpt2-medium")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32 if self.device == "cpu" else torch.float16,
        }
        
        # Load the fine-tuned model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            **model_kwargs
        )
        
        # Move to device
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_response(self, prompt, max_length=200, temperature=0.7, top_p=0.9, do_sample=True):
        """Generate a response to the given prompt"""
        
        # Format prompt (match training format)
        formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        )
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_return_sequences=1
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1].strip()
        else:
            response = full_response[len(formatted_prompt):].strip()
        
        return response
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n=== Interactive Chat ===")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'clear' to clear conversation history")
        print("-" * 40)
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    conversation_history = []
                    print("Conversation history cleared!")
                    continue
                
                if not user_input:
                    continue
                
                # Generate response
                print("Assistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
                # Add to history
                conversation_history.append(f"You: {user_input}")
                conversation_history.append(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError generating response: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./fine_tuned_model/final_model",
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        help="Single prompt to generate response for (non-interactive mode)"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=200,
        help="Maximum length of generated response"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature (0.1 = more focused, 1.0 = more creative)"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda", "mps"],
        help="Device to use for inference (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path '{model_path}' does not exist!")
        print("Make sure you've trained a model first using finetuning.py")
        return
    
    # Initialize inference
    try:
        inference = ModelInference(model_path, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if args.prompt:
        # Single prompt mode
        print(f"Prompt: {args.prompt}")
        print("Generating response...")
        
        response = inference.generate_response(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        print(f"\nResponse: {response}")
    else:
        # Interactive mode
        inference.interactive_chat()


if __name__ == "__main__":
    main()