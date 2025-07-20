#!/usr/bin/env python3
"""
Test system message filtering functionality
"""

import sys
import os

# Add the current directory to the path so we can import finetuning
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finetuning import MessengerDataProcessor

def test_system_message_filtering():
    """Test that system messages are correctly identified and filtered"""
    
    processor = MessengerDataProcessor("dummy_path")
    
    # Test English system messages
    english_test_cases = [
        ("Alice has left the group", True),
        ("Bob joined the conversation", True), 
        ("Charlie liked your message", True),
        ("Diana sent a photo", True),
        ("Eve started a call", True),
        ("Frank is now an admin", True),
        ("The group was created", True),
        ("Hello everyone!", False),
        ("How are you doing?", False),
        ("Let's meet tomorrow", False),
    ]
    
    # Test French system messages
    french_test_cases = [
        ("Alice a quitté le groupe", True),
        ("Bob a rejoint la conversation", True),
        ("Charlie a aimé votre message", True),
        ("Diana a envoyé une photo", True),
        ("Eve a commencé un appel", True),
        ("Frank est maintenant administrateur", True),
        ("Le groupe a été créé", True),
        ("Bonjour tout le monde!", False),
        ("Comment allez-vous?", False),
        ("Rendez-vous demain", False),
    ]
    
    print("Testing English system message detection:")
    for message, expected in english_test_cases:
        result = processor.is_system_message(message)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{message}' -> {result} (expected {expected})")
    
    print("\nTesting French system message detection:")
    for message, expected in french_test_cases:
        result = processor.is_system_message(message)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{message}' -> {result} (expected {expected})")
    
    # Test edge cases
    print("\nTesting edge cases:")
    edge_cases = [
        ("", True),  # Empty string
        (None, True),  # None value
        ("   ", False),  # Whitespace only
        ("John liked your message", True),  # Should match exact system message
        ("I really liked your message", False),  # Should not match as regular message
    ]
    
    for message, expected in edge_cases:
        result = processor.is_system_message(message)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{message}' -> {result} (expected {expected})")

if __name__ == "__main__":
    test_system_message_filtering()