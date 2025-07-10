import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

# Load your trained model here (copy your model class definition)
# [Your GPTLanguageModel class goes here]


class ChatBot:
    def __init__(self, model_path: str, vocab_path: str = None):
        """
        Initialize chatbot with trained model

        Args:
            model_path: Path to saved model state dict
            vocab_path: Path to vocabulary pickle file (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load vocabulary
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                vocab_data = pickle.load(f)
                self.stoi = vocab_data["stoi"]
                self.itos = vocab_data["itos"]
        else:
            # If no vocab file, you'll need to recreate from training data
            print(
                "Warning: No vocabulary file found. You'll need to recreate it from your training data."
            )
            self.stoi = None
            self.itos = None

        # Load model
        self.model = GPTLanguageModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Special tokens
        self.msg_token = "<MSG>"
        self.you_token = "<YOU>"
        self.end_token = "<END>"

    def encode(self, text: str):
        """Encode text to token indices"""
        if self.stoi is None:
            raise ValueError(
                "Vocabulary not loaded. Please provide vocab_path or recreate vocabulary."
            )
        return [self.stoi.get(c, self.stoi.get("<UNK>", 0)) for c in text]

    def decode(self, tokens):
        """Decode token indices to text"""
        if self.itos is None:
            raise ValueError(
                "Vocabulary not loaded. Please provide vocab_path or recreate vocabulary."
            )
        return "".join([self.itos.get(i, "<UNK>") for i in tokens])

    def clean_response(self, response: str) -> str:
        """Clean and post-process the generated response"""
        # Remove special tokens from response
        response = response.replace(self.msg_token, "")
        response = response.replace(self.you_token, "")
        response = response.replace(self.end_token, "")

        # Remove excessive whitespace
        response = " ".join(response.split())

        # Truncate at sentence boundaries if too long
        if len(response) > 200:
            # Find last sentence ending
            for punct in [".", "!", "?"]:
                last_punct = response.rfind(punct)
                if last_punct > 100:  # Ensure we don't cut too short
                    response = response[: last_punct + 1]
                    break

        return response.strip()

    def generate_response(
        self, input_message: str, max_new_tokens: int = 150, temperature: float = 0.8
    ) -> str:
        """
        Generate a response to an input message

        Args:
            input_message: The message to respond to
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
        """
        # Format input with special tokens
        prompt = f"{self.msg_token}{input_message}{self.you_token}"

        # Encode the prompt
        encoded_prompt = self.encode(prompt)
        context = torch.tensor([encoded_prompt], dtype=torch.long, device=self.device)

        # Generate response
        with torch.no_grad():
            generated = self.model.generate(context, max_new_tokens)
            generated_text = self.decode(generated[0].tolist())

        # Extract just the response part
        response_start = generated_text.find(self.you_token)
        if response_start != -1:
            response = generated_text[response_start + len(self.you_token) :]
            # Stop at end token
            end_pos = response.find(self.end_token)
            if end_pos != -1:
                response = response[:end_pos]
            # Stop at next message token (in case model generates multiple exchanges)
            next_msg_pos = response.find(self.msg_token)
            if next_msg_pos != -1:
                response = response[:next_msg_pos]
        else:
            response = generated_text

        return self.clean_response(response)

    def chat_loop(self):
        """Interactive chat loop"""
        print("Chatbot loaded! Type 'quit' to exit.")
        print("=" * 50)

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            try:
                response = self.generate_response(user_input)
                print(f"Bot: {response}")
            except Exception as e:
                print(f"Error generating response: {e}")


# Utility function to save vocabulary during training
def save_vocabulary(text_file: str, vocab_file: str):
    """Save vocabulary from training text file"""
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    vocab_data = {"stoi": stoi, "itos": itos, "vocab_size": len(chars)}

    with open(vocab_file, "wb") as f:
        pickle.dump(vocab_data, f)

    print(f"Vocabulary saved to {vocab_file}")
    print(f"Vocabulary size: {len(chars)}")


# Example usage
if __name__ == "__main__":
    # First, save vocabulary from your training data
    save_vocabulary("input.txt", "vocabulary.pkl")

    # Then create and use the chatbot
    chatbot = ChatBot("model.pth", "vocabulary.pkl")
    chatbot.chat_loop()
