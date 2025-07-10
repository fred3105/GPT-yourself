import os
import json
import re
import unicodedata
from typing import Literal
from typing import List, Dict, Any


class MessengerDataExtractor:
    def __init__(
        self,
        data_folder_path: str,
        your_name: str,
        format_type: Literal["char_level", "structured"] = "char_level",
    ):
        """
        Initialize the extractor with the path to your exported data folder
        and your name as it appears in the messages.

        Args:
            data_folder_path: Path to the root folder of exported messenger data
            your_name: Your name as it appears in the messenger data
            format_type: "char_level" for character-based GPT, "structured" for token-based
        """
        self.data_folder_path = data_folder_path
        self.your_name = your_name
        self.format_type = format_type
        self.message_pairs = []

    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        # Pattern to match emojis
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002702-\U000027b0"  # dingbats
            "\U000024c2-\U0001f251"  # enclosed characters
            "\U0001f900-\U0001f9ff"  # supplemental symbols
            "\U0001fa70-\U0001faff"  # symbols and pictographs extended-a
            "\U00002600-\U000026ff"  # miscellaneous symbols
            "\U00002700-\U000027bf"  # dingbats
            "\U0001f018-\U0001f270"  # various symbols
            "\U00001f00-\U00001f6f"  # additional emojis
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text)

    def is_system_message(self, content: str) -> bool:
        """Check if the message is a system message or reaction."""
        # French system message patterns
        system_patterns = [
            r"a rejoint l'appel",
            r"a quitté l'appel",
            r"Vous avez rejoint l'appel",
            r"Vous avez quitté l'appel",
            r"a rejoint le groupe",
            r"a quitté le groupe",
            r"a ajouté .+ au groupe",
            r"a retiré .+ du groupe",
            r"a changé le nom du groupe",
            r"a changé la photo du groupe",
            r"a reagi .+ a votre message",
            r"a reagi .+ au message",
            r"Vous avez reagi .+ au message",
            r"a partagé un lien",
            r"a partagé une photo",
            r"a partagé une vidéo",
            r"Appel manqué",
            r"Appel terminé",
            r"a créé le groupe",
            r"Messages chiffrés de bout en bout",
            r"est maintenant connecté",
            r"était actif",
            r"a modifié les paramètres",
            r"a activé",
            r"a désactivé",
        ]

        # English system message patterns (in case there are mixed languages)
        english_patterns = [
            r"joined the call",
            r"left the call",
            r"You joined the call",
            r"You left the call",
            r"joined the group",
            r"left the group",
            r"added .+ to the group",
            r"removed .+ from the group",
            r"changed the group name",
            r"changed the group photo",
            r"reacted .+ to your message",
            r"reacted .+ to the message",
            r"You reacted .+ to the message",
            r"shared a link",
            r"shared a photo",
            r"shared a video",
            r"Missed call",
            r"Call ended",
            r"created the group",
            r"End-to-end encrypted",
            r"is now connected",
            r"was active",
            r"changed the settings",
            r"enabled",
            r"disabled",
        ]

        all_patterns = system_patterns + english_patterns

        for pattern in all_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Fix Facebook's double-encoded UTF-8 issues
        try:
            text = text.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

        # Remove emojis first
        text = self.remove_emojis(text)

        # Remove @ symbols (mentions)
        text = re.sub(r"@\w+", "", text)

        # Remove standalone @ symbols
        text = re.sub(r"@", "", text)

        # Normalize Unicode characters to basic ASCII equivalents
        text = unicodedata.normalize("NFD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")

        # Additional manual replacements for common cases
        replacements = {
            "à": "a",
            "á": "a",
            "â": "a",
            "ã": "a",
            "ä": "a",
            "å": "a",
            "è": "e",
            "é": "e",
            "ê": "e",
            "ë": "e",
            "ì": "i",
            "í": "i",
            "î": "i",
            "ï": "i",
            "ò": "o",
            "ó": "o",
            "ô": "o",
            "õ": "o",
            "ö": "o",
            "ù": "u",
            "ú": "u",
            "û": "u",
            "ü": "u",
            "ý": "y",
            "ÿ": "y",
            "ñ": "n",
            "ç": "c",
            "À": "A",
            "Á": "A",
            "Â": "A",
            "Ã": "A",
            "Ä": "A",
            "Å": "A",
            "È": "E",
            "É": "E",
            "Ê": "E",
            "Ë": "E",
            "Ì": "I",
            "Í": "I",
            "Î": "I",
            "Ï": "I",
            "Ò": "O",
            "Ó": "O",
            "Ô": "O",
            "Õ": "O",
            "Ö": "O",
            "Ù": "U",
            "Ú": "U",
            "Û": "U",
            "Ü": "U",
            "Ý": "Y",
            "Ÿ": "Y",
            "Ñ": "N",
            "Ç": "C",
            '"': '"',
            '"': '"',
            """: "'", """: "'",
            "–": "-",
            "—": "-",
            "…": "...",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove control characters
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

        return text.strip()

    def extract_messages_from_json(self, json_file_path: str) -> List[Dict[str, Any]]:
        """Extract messages from a single JSON file."""
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            messages = []
            if "messages" in data:
                for message in data["messages"]:
                    if "content" not in message or "sender_name" not in message:
                        continue

                    # Skip call duration messages
                    if "call_duration" in message:
                        continue

                    raw_content = message["content"]
                    if not raw_content or not raw_content.strip():
                        continue

                    # Check if it's a system message before cleaning
                    if self.is_system_message(raw_content):
                        continue

                    sender_name = self.clean_text(message["sender_name"])
                    content = self.clean_text(raw_content)

                    # Skip if content is empty after cleaning
                    if not content:
                        continue

                    # Skip if content is too short (likely just punctuation or artifacts)
                    if len(content) < 2:
                        continue

                    messages.append(
                        {
                            "sender": sender_name,
                            "content": content,
                            "timestamp": message.get("timestamp_ms", 0),
                        }
                    )

            # Sort messages by timestamp (oldest first)
            messages.sort(key=lambda x: x["timestamp"])
            return messages

        except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
            print(f"Error reading {json_file_path}: {e}")
            return []

    def find_your_messages_with_context(
        self, messages: List[Dict[str, Any]]
    ) -> List[tuple]:
        """Find your messages along with the previous message for context."""
        message_pairs = []

        for i, message in enumerate(messages):
            if (
                message["sender"] == self.your_name
                and i > 0
                and len(message["content"]) > 0
            ):
                previous_message = messages[i - 1]

                if (
                    previous_message["sender"] != self.your_name
                    and len(previous_message["content"]) > 0
                ):
                    message_pairs.append(
                        (previous_message["content"], message["content"])
                    )

        return message_pairs

    def process_all_files(self):
        """Process all JSON files in the data folder."""
        total_files = 0
        processed_files = 0
        skipped_system_messages = 0

        print(f"Starting to process files in: {self.data_folder_path}")

        for root, dirs, files in os.walk(self.data_folder_path):
            for file in files:
                if file.endswith(".json"):
                    total_files += 1
                    file_path = os.path.join(root, file)

                    print(f"Processing: {file_path}")

                    messages = self.extract_messages_from_json(file_path)
                    if messages:
                        pairs = self.find_your_messages_with_context(messages)
                        self.message_pairs.extend(pairs)
                        processed_files += 1
                        print(f"  Found {len(pairs)} message pairs")

        print(f"\nProcessed {processed_files} of {total_files} JSON files")
        print(f"Total message pairs extracted: {len(self.message_pairs)}")

    def save_training_data(self, output_file: str = "input.txt"):
        """Save the extracted message pairs to a text file."""

        if not self.message_pairs:
            print(
                "No message pairs found. Make sure your name is correct and you have conversations."
            )
            return

        with open(output_file, "w", encoding="utf-8") as f:
            if self.format_type == "char_level":
                # Format for character-level GPT (continuous text)
                for i, (context, response) in enumerate(self.message_pairs):
                    # Use special tokens to mark conversation boundaries
                    f.write(f"<MSG>{context}<YOU>{response}<END>")
                    if i < len(self.message_pairs) - 1:
                        f.write("\n")
            else:
                # Structured format for token-based models
                for i, (context, response) in enumerate(self.message_pairs):
                    f.write(f"Input: {context}\n")
                    f.write(f"Output: {response}\n")
                    f.write("---\n")

        print(f"\nTraining data saved to: {output_file}")
        print(f"Total examples: {len(self.message_pairs)}")
        print(f"Format: {self.format_type}")

        # Show some statistics
        if self.format_type == "char_level":
            total_chars = sum(len(f"{pair[0]}{pair[1]}") for pair in self.message_pairs)
            print(f"Total characters: {total_chars:,}")
            print(
                f"Average characters per example: {total_chars / len(self.message_pairs):.1f}"
            )
        else:
            avg_context_length = sum(len(pair[0]) for pair in self.message_pairs) / len(
                self.message_pairs
            )
            avg_response_length = sum(
                len(pair[1]) for pair in self.message_pairs
            ) / len(self.message_pairs)
            print(f"Average context length: {avg_context_length:.1f} characters")
            print(f"Average response length: {avg_response_length:.1f} characters")

    def preview_data(self, num_examples: int = 3):
        """Preview some of the extracted message pairs."""
        if not self.message_pairs:
            print("No message pairs found.")
            return

        print(
            f"\nPreview of {min(num_examples, len(self.message_pairs))} message pairs:"
        )
        print("=" * 60)

        for i, (context, response) in enumerate(self.message_pairs[:num_examples]):
            print(f"\nExample {i + 1}:")
            if self.format_type == "char_level":
                print(f"Formatted: <MSG>{context}<YOU>{response}<END>")
            else:
                print(f"Context: {context}")
                print(f"Your response: {response}")
            print("-" * 40)


def main():
    # Configuration - UPDATE THESE VALUES
    DATA_FOLDER = "data"  # Path to your exported messenger data folder
    YOUR_NAME = "Frederic Legrand"  # Your name as it appears in the messages

    # Choose format type based on your model
    # "char_level" for your character-based GPT
    # "structured" for token-based models
    FORMAT_TYPE = "char_level"

    OUTPUT_FILE = "input.txt"  # This matches your GPT code

    # Create extractor instance
    extractor = MessengerDataExtractor(DATA_FOLDER, YOUR_NAME, FORMAT_TYPE)

    # Process all files
    extractor.process_all_files()

    # Preview some examples
    extractor.preview_data(num_examples=3)

    # Save training data
    extractor.save_training_data(OUTPUT_FILE)

    print(f"\nDone! Your training data is ready in '{OUTPUT_FILE}'")
    print(f"Format optimized for: {FORMAT_TYPE} model")
    if FORMAT_TYPE == "char_level":
        print(
            "Special tokens used: <MSG> (incoming message), <YOU> (your response), <END> (end of conversation)"
        )
    print("\nFiltering applied:")
    print("- Removed emojis")
    print("- Removed @ symbols and mentions")
    print("- Filtered out system messages and reactions")
    print("- Skipped messages shorter than 2 characters")


if __name__ == "__main__":
    main()
