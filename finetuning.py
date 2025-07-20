#!/usr/bin/env python3
"""
Fine-tune an LLM on Facebook Messenger data - macOS Compatible Version
Uses MPS acceleration and alternative quantization methods
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import platform
import warnings
import os

# Core imports - no bitsandbytes
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.training_args import TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import wandb

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)


class MessengerDataProcessor:
    """Process Facebook Messenger data into training format"""

    def __init__(self, data_path, min_conversation_length=2, max_context_length=2048):
        self.data_path = Path(data_path)
        self.min_conversation_length = min_conversation_length
        self.max_context_length = max_context_length
        self.conversations = []

    def load_messenger_data(self):
        """Load Facebook Messenger JSON data"""
        print(f"Loading data from {self.data_path}...")

        # Facebook data is usually in multiple JSON files
        if self.data_path.is_dir():
            json_files = list(self.data_path.glob("**/*.json"))
            all_messages = []

            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "messages" in data:
                            all_messages.extend(data["messages"])
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        else:
            # Single file
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_messages = data.get("messages", [])

        print(f"Loaded {len(all_messages)} messages")
        return all_messages

    def create_conversations(self, messages):
        """Group messages into conversations"""
        # Sort messages by timestamp
        messages.sort(key=lambda x: x.get("timestamp_ms", 0))

        conversations = []
        current_conv = []
        last_timestamp = 0

        # Group messages into conversations (30 min gap = new conversation)
        for msg in messages:
            timestamp = msg.get("timestamp_ms", 0)

            # If more than 30 minutes since last message, start new conversation
            if timestamp - last_timestamp > 30 * 60 * 1000:
                if len(current_conv) >= self.min_conversation_length:
                    conversations.append(current_conv)
                current_conv = []

            if "content" in msg and msg["content"]:
                current_conv.append(msg)
                last_timestamp = timestamp

        # Add last conversation
        if len(current_conv) >= self.min_conversation_length:
            conversations.append(current_conv)

        print(f"Created {len(conversations)} conversations")
        return conversations

    def create_training_pairs(self, conversations, your_name=None):
        """Create prompt-response pairs from conversations"""
        training_data = []

        for conv in conversations:
            # Create pairs from consecutive messages
            for i in range(len(conv) - 1):
                # If your_name is specified, only use your messages as responses
                if your_name:
                    if conv[i + 1].get("sender_name") == your_name:
                        prompt = conv[i].get("content", "")
                        response = conv[i + 1].get("content", "")

                        if prompt and response:
                            training_data.append(
                                {
                                    "prompt": prompt,
                                    "response": response,
                                    "context": self._get_context(conv, i),
                                }
                            )
                else:
                    # Use all message pairs
                    prompt = conv[i].get("content", "")
                    response = conv[i + 1].get("content", "")

                    if prompt and response:
                        training_data.append(
                            {
                                "prompt": prompt,
                                "response": response,
                                "context": self._get_context(conv, i),
                            }
                        )

        print(f"Created {len(training_data)} training pairs")
        return training_data

    def _get_context(self, conversation, current_idx, context_window=3):
        """Get conversation context (previous messages)"""
        start_idx = max(0, current_idx - context_window)
        context_msgs = []

        for i in range(start_idx, current_idx):
            sender = conversation[i].get("sender_name", "Unknown")
            content = conversation[i].get("content", "")
            if content:
                context_msgs.append(f"{sender}: {content}")

        return "\n".join(context_msgs)

    def prepare_dataset(self, your_name=None):
        """Main method to prepare the dataset"""
        messages = self.load_messenger_data()
        conversations = self.create_conversations(messages)
        training_data = self.create_training_pairs(conversations, your_name)

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(training_data)

        # Filter out very long conversations
        df["total_length"] = df["prompt"].str.len() + df["response"].str.len()
        df = df[df["total_length"] < self.max_context_length]

        print(f"Final dataset size: {len(df)} examples")
        return df


class MacOSLLMFineTuner:
    """Fine-tune LLM on macOS using LoRA and MPS acceleration"""

    def __init__(
        self,
        base_model="Qwen/Qwen-1_8B",  # Qwen model for fine-tuning
        max_seq_length=2048,
        output_dir="./fine_tuned_model",
        use_mps=None,
    ):
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Detect device
        if use_mps is None:
            self.device = self._detect_device()
        else:
            self.device = "mps" if use_mps else "cpu"

        print(f"Using device: {self.device}")

    def _detect_device(self):
        """Detect the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Check if MPS is actually built
            try:
                # Try to create a tensor on MPS
                test_tensor = torch.zeros(1).to("mps")
                del test_tensor
                return "mps"
            except Exception as e:
                print(f"MPS available but not functional: {e}")
                return "cpu"
        else:
            return "cpu"

    def load_model(self, load_in_8bit=False):
        """Load the base model with optional 8-bit quantization"""
        print(f"Loading model: {self.base_model}")

        # Model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32 if self.device == "cpu" else torch.float16,
        }

        # For macOS, we can't use bitsandbytes, but we can use torch's native quantization
        if load_in_8bit and self.device != "cpu":
            print(
                "Note: 8-bit quantization is limited on macOS. Using float16 instead."
            )
            model_kwargs["torch_dtype"] = torch.float16

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True, padding_side="right"
        )

        # Add padding token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs)

        # Move to device
        if self.device != "cpu":
            model = model.to(self.device)

        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()

        # Apply LoRA
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=[
                "c_attn",
                "c_proj",
            ],  # You might need to adjust these based on the model architecture
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    def format_dataset(self, df, tokenizer):
        """Format dataset for training"""
        # Convert to the format expected by the model
        formatted_data = []

        for _, row in df.iterrows():
            # Create conversation format
            if row["context"]:
                conversation = f"Previous conversation:\n{row['context']}\n\n"
            else:
                conversation = ""

            # Simple format for training
            text = f"{conversation}User: {row['prompt']}\nAssistant: {row['response']}"

            formatted_data.append({"text": text})

        # Create HuggingFace dataset
        dataset = Dataset.from_list(formatted_data)

        # Add tokenization to the dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt",
            )

        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def train(
        self, dataset, model, tokenizer, num_epochs=3, batch_size=4, wandb_enabled=False
    ):
        """Train the model"""
        print("Starting training...")

        # Training arguments optimized for macOS
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            num_train_epochs=num_epochs,
            learning_rate=2e-5,
            warmup_steps=0,
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            eval_strategy="no",
            save_total_limit=3,
            load_best_model_at_end=False,
            optim="adamw_torch",  # Standard optimizer for macOS
            fp16=False,  # FP16 is not supported on macOS
            bf16=False,  # BF16 is not supported on macOS
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            dataloader_num_workers=0,  # Important for macOS
            remove_unused_columns=False,
            report_to="wandb" if wandb_enabled else "none",
        )

        # Data collator for causal language modeling
        from transformers.data.data_collator import DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # For causal LM, not masked LM
        )

        # Create trainer using standard Trainer instead of SFTTrainer
        from transformers.trainer import Trainer

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Log final metrics to wandb if enabled
        if wandb_enabled:
            try:
                # Log training metrics
                train_loss = trainer.state.log_history[-1].get("train_loss", 0)
                wandb.log(
                    {
                        "final_train_loss": train_loss,
                        "total_training_steps": trainer.state.global_step,
                    }
                )

                # Log model parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                wandb.log(
                    {
                        "total_parameters": total_params,
                        "trainable_parameters": trainable_params,
                        "trainable_percentage": (trainable_params / total_params) * 100,
                    }
                )
            except Exception as e:
                print(f"Warning: Could not log final metrics to wandb: {e}")

        return trainer

    def save_model(self, model, tokenizer, save_path=None):
        """Save the fine-tuned model"""
        if save_path is None:
            save_path = self.output_dir / "final_model"

        print(f"Saving model to {save_path}")

        # Save the model
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        # Save configuration
        config = {
            "base_model": self.base_model,
            "max_seq_length": self.max_seq_length,
            "device": self.device,
            "lora_config": model.peft_config,
        }

        import json

        with open(save_path / "training_config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        print("Model saved successfully!")


def estimate_training_time(num_examples, device="mps", model_size="3b"):
    """Estimate training time based on hardware and dataset size"""
    # Rough estimates for macOS
    if device == "mps":  # Apple Silicon
        base_rate = {
            "phi-2": 0.3,  # hours per 10k examples
            "qwen-1_8b": 0.5,  # hours per 10k examples
            "3b": 0.8,
            "7b": 2.0,
        }
    else:  # CPU
        base_rate = {
            "phi-2": 2.0,  # hours per 10k examples
            "qwen-1_8b": 3.0,  # hours per 10k examples
            "3b": 5.0,
            "7b": 12.0,
        }

    hours = (num_examples / 10000) * base_rate.get(model_size, 1.0)
    return hours


def check_environment():
    """Check the macOS environment"""
    print("=== Environment Check ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    # Check for Apple Silicon
    if platform.processor() == "arm":
        print("Apple Silicon detected")
    else:
        print("Intel Mac detected")

    # Check MPS availability
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available")
        try:
            test = torch.zeros(1).to("mps")
            print("MPS is functional")
            del test
        except Exception as e:
            print(f"MPS is not functional: {e}")
    else:
        print("MPS is not available")

    print("=" * 30 + "\n")


def wandb_login():
    """Handle Weights & Biases login"""
    try:
        # Check if user is already logged in
        if wandb.api.api_key:
            print("Already logged into wandb")
            return True
    except Exception:
        pass

    # Try to login using environment variable or prompt
    try:
        wandb.login()
        print("Successfully logged into wandb")
        return True
    except Exception as e:
        print(f"Warning: Could not login to wandb: {e}")
        print("You can login manually by running: wandb login")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM on Facebook Messenger data (macOS)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to Facebook Messenger data (JSON file or directory)",
    )
    parser.add_argument(
        "--your_name",
        type=str,
        default=None,
        help="Your name in conversations (to filter only your responses)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen-1_8B",
        help="Base model to fine-tune (Qwen-1_8B recommended for macOS)",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fine_tuned_model",
        help="Output directory for the model",
    )
    parser.add_argument(
        "--use_cpu", action="store_true", help="Force CPU usage instead of MPS"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llm-finetuning",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Weights & Biases run name"
    )
    parser.add_argument(
        "--disable_wandb", action="store_true", help="Disable Weights & Biases tracking"
    )

    args = parser.parse_args()

    # Initialize wandb if not disabled
    wandb_enabled = False
    if not args.disable_wandb:
        wandb_enabled = wandb_login()
        if wandb_enabled:
            # Initialize wandb run
            run_name = (
                args.wandb_run_name
                or f"{args.model.split('/')[-1]}-{args.your_name or 'general'}"
            )
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model": args.model,
                    "max_seq_length": args.max_seq_length,
                    "num_epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "your_name": args.your_name,
                    "data_path": args.data_path,
                    "use_cpu": args.use_cpu,
                },
            )

    # Check environment
    check_environment()

    # Process data
    print("Processing Facebook Messenger data...")
    processor = MessengerDataProcessor(
        args.data_path, max_context_length=args.max_seq_length
    )
    df = processor.prepare_dataset(your_name=args.your_name)

    # Log dataset info to wandb if enabled
    if wandb_enabled:
        wandb.log(
            {
                "dataset_size": len(df),
                "avg_prompt_length": df["prompt"].str.len().mean(),
                "avg_response_length": df["response"].str.len().mean(),
            }
        )

    # Estimate training time
    device = "cpu" if args.use_cpu else "mps"
    est_hours = estimate_training_time(len(df), device=device, model_size="qwen-1_8b")
    print(f"\nEstimated training time: {est_hours:.1f} hours")

    # Initialize fine-tuner
    fine_tuner = MacOSLLMFineTuner(
        base_model=args.model,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        use_mps=not args.use_cpu,
    )

    # Load model
    model, tokenizer = fine_tuner.load_model()

    # Format dataset
    dataset = fine_tuner.format_dataset(df, tokenizer)

    # Train
    fine_tuner.train(
        dataset,
        model,
        tokenizer,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        wandb_enabled=wandb_enabled,
    )

    # Save model
    fine_tuner.save_model(model, tokenizer)

    # Finish wandb run if enabled
    if wandb_enabled:
        wandb.finish()

    print("\nTraining complete! Your model is saved in:", args.output_dir)
    print("\nTo test your model, run:")
    print(f"python test_model_macos.py --model_path {args.output_dir}/final_model")


if __name__ == "__main__":
    main()
