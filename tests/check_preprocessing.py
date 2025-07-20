#!/usr/bin/env python3
"""
Script to check preprocessing quality by loading and displaying random conversations
"""

import json
import random
import tempfile
from pathlib import Path
from finetuning import MessengerDataProcessor
import glob
import os


def get_real_message_files(data_dir="/Users/fredericlegrand/Documents/GitHub/ng-video-lecture/data"):
    """Get a list of real message files from the data directory"""
    message_files = []
    
    # Search for all message_1.json files in subdirectories
    pattern = os.path.join(data_dir, "messages", "**", "message_1.json")
    message_files = glob.glob(pattern, recursive=True)
    
    return message_files


def load_random_real_conversation(data_dir="/Users/fredericlegrand/Documents/GitHub/ng-video-lecture/data"):
    """Load a random real conversation from the data directory"""
    message_files = get_real_message_files(data_dir)
    
    if not message_files:
        raise ValueError(f"No message files found in {data_dir}")
    
    # Pick a random conversation
    random_file = random.choice(message_files)
    print(f"Loading conversation from: {random_file}")
    
    try:
        with open(random_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data, random_file
    except Exception as e:
        print(f"Error loading {random_file}: {e}")
        # Try another file if this one fails
        message_files.remove(random_file)
        if message_files:
            return load_random_real_conversation(data_dir)
        else:
            raise ValueError("No readable message files found")


def display_preprocessing_comparison(processor, messages):
    """Display before/after preprocessing comparison"""
    print("=" * 80)
    print("PREPROCESSING COMPARISON")
    print("=" * 80)
    
    for i, msg in enumerate(messages[:8]):  # Show first 8 messages
        original = msg.get("content", "")
        processed = processor.preprocess_text(original)
        
        print(f"\nMessage {i+1} ({msg.get('sender_name', 'Unknown')}):")
        print(f"  BEFORE: {repr(original)}")
        print(f"  AFTER:  {repr(processed)}")
        
        # Highlight what was removed
        removed_items = []
        if "😀" in original or "🎉" in original or "💯" in original:
            removed_items.append("emojis")
        if "http" in original:
            removed_items.append("URLs")
        if "@" in original:
            removed_items.append("emails")
        
        if removed_items:
            print(f"  REMOVED: {', '.join(removed_items)}")
        
        if not processed:
            print("  ⚠️  MESSAGE FILTERED OUT (no content after preprocessing)")


def display_conversations(conversations):
    """Display extracted conversations"""
    print("\n" + "=" * 80)
    print("EXTRACTED CONVERSATIONS")
    print("=" * 80)
    
    for i, conv in enumerate(conversations):
        print(f"\n--- Conversation {i+1} ({len(conv)} messages) ---")
        for j, msg in enumerate(conv):
            sender = msg.get("sender_name", "Unknown")
            content = msg.get("content", "")
            print(f"  {j+1}. {sender}: {content}")


def display_training_pairs(training_data, max_pairs=5):
    """Display training pairs"""
    print("\n" + "=" * 80)
    print("TRAINING PAIRS (showing first {})".format(min(max_pairs, len(training_data))))
    print("=" * 80)
    
    for i, pair in enumerate(training_data[:max_pairs]):
        print(f"\n--- Training Pair {i+1} ---")
        print(f"PROMPT:   {pair['prompt']}")
        print(f"RESPONSE: {pair['response']}")
        if pair['context']:
            print(f"CONTEXT:\n{pair['context']}")


def check_preprocessing_quality(training_data):
    """Check the quality of preprocessing"""
    print("\n" + "=" * 80)
    print("PREPROCESSING QUALITY CHECK")
    print("=" * 80)
    
    issues = []
    total_text = ""
    
    for pair in training_data:
        total_text += pair['prompt'] + " " + pair['response'] + " "
    
    # Check for remaining problematic content
    if "😀" in total_text or "🎉" in total_text or "💯" in total_text:
        issues.append("❌ Emojis still present")
    else:
        print("✅ No emojis found")
    
    if "http://" in total_text or "https://" in total_text:
        issues.append("❌ URLs still present") 
    else:
        print("✅ No URLs found")
        
    if "@" in total_text and ".com" in total_text:
        issues.append("❌ Email addresses still present")
    else:
        print("✅ No email addresses found")
    
    # Check for empty content
    empty_prompts = sum(1 for pair in training_data if not pair['prompt'].strip())
    empty_responses = sum(1 for pair in training_data if not pair['response'].strip())
    
    if empty_prompts > 0:
        issues.append(f"❌ {empty_prompts} empty prompts")
    else:
        print("✅ No empty prompts")
        
    if empty_responses > 0:
        issues.append(f"❌ {empty_responses} empty responses")
    else:
        print("✅ No empty responses")
    
    # Summary
    print(f"\nDataset size: {len(training_data)} training pairs")
    avg_prompt_len = sum(len(pair['prompt']) for pair in training_data) / len(training_data)
    avg_response_len = sum(len(pair['response']) for pair in training_data) / len(training_data)
    print(f"Average prompt length: {avg_prompt_len:.1f} characters")
    print(f"Average response length: {avg_response_len:.1f} characters")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n🎉 All preprocessing checks passed!")


def main():
    """Main function to demonstrate preprocessing"""
    print("FACEBOOK MESSENGER PREPROCESSING CHECKER")
    print("=" * 80)
    
    # Load real data instead of sample data
    try:
        real_data, source_file = load_random_real_conversation()
        print(f"\nUsing real conversation data from: {Path(source_file).parent.name}")
        print(f"Total messages in conversation: {len(real_data.get('messages', []))}")
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(real_data, f, indent=2, ensure_ascii=False)
            temp_path = f.name
    
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Falling back to sample data...")
        # Fallback to sample data if real data fails
        sample_data = {
            "messages": [
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1000000,
                    "content": "Hey! 😀 This is sample data since real data couldn't be loaded."
                },
                {
                    "sender_name": "Bob", 
                    "timestamp_ms": 1000100,
                    "content": "Got it! Sample conversation here."
                }
            ]
        }
        real_data = sample_data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(real_data, f, indent=2)
            temp_path = f.name
    
    try:
        # Initialize processor
        processor = MessengerDataProcessor(temp_path, min_conversation_length=2)
        
        # Show preprocessing comparison
        display_preprocessing_comparison(processor, real_data["messages"])
        
        # Load and process data
        messages = processor.load_messenger_data()
        conversations = processor.create_conversations(messages)
        training_data = processor.create_training_pairs(conversations)
        
        # Display results
        display_conversations(conversations)
        display_training_pairs(training_data)
        check_preprocessing_quality(training_data)
        
        # Show random samples
        if len(training_data) > 3:
            print("\n" + "=" * 80)
            print("RANDOM SAMPLES")
            print("=" * 80)
            random_pairs = random.sample(training_data, min(3, len(training_data)))
            for i, pair in enumerate(random_pairs):
                print(f"\n--- Random Sample {i+1} ---")
                print(f"PROMPT:   {pair['prompt']}")
                print(f"RESPONSE: {pair['response']}")
                
    finally:
        # Cleanup
        Path(temp_path).unlink()


if __name__ == "__main__":
    main()