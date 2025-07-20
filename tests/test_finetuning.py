#!/usr/bin/env python3
"""
Pytest tests for finetuning.py - Facebook Messenger data processing and preprocessing
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from finetuning import MessengerDataProcessor, MacOSLLMFineTuner


class TestMessengerDataProcessor:
    """Test suite for MessengerDataProcessor class"""

    @pytest.fixture
    def processor(self):
        """Create a MessengerDataProcessor instance for testing"""
        return MessengerDataProcessor("dummy_path", min_conversation_length=2, max_context_length=1024)

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing"""
        return [
            {
                "sender_name": "Alice",
                "timestamp_ms": 1000000,
                "content": "Hello! 😀 How are you?"
            },
            {
                "sender_name": "Bob", 
                "timestamp_ms": 1000001,
                "content": "I'm good! Check this: https://example.com 🎉"
            },
            {
                "sender_name": "Alice",
                "timestamp_ms": 1002000,  # 2 seconds later
                "content": "Cool link! Email me at alice@test.com"
            },
            {
                "sender_name": "Bob",
                "timestamp_ms": 3000000,  # 33+ minutes later (new conversation)
                "content": "Starting new topic 👍"
            }
        ]

    @pytest.fixture
    def sample_json_data(self, sample_messages):
        """Sample JSON data structure"""
        return {"messages": sample_messages}

    def test_preprocess_text_removes_emojis(self, processor):
        """Test that emojis are removed from text"""
        test_cases = [
            ("Hello 😀 world 🎉", "Hello world"),
            ("No emojis here", "No emojis here"),
            ("🔥🎯💯", ""),
            ("Mixed 😊 content 🚀 here", "Mixed content here")
        ]
        
        for input_text, expected in test_cases:
            result = processor.preprocess_text(input_text)
            assert expected in result or (expected == "" and result == "")

    def test_preprocess_text_removes_urls(self, processor):
        """Test that URLs are removed from text"""
        test_cases = [
            ("Check https://example.com out", "Check out"),
            ("Visit http://test.org please", "Visit please"),
            ("Multiple https://first.com and http://second.org links", "Multiple and links"),
            ("No URLs here", "No URLs here")
        ]
        
        for input_text, expected in test_cases:
            result = processor.preprocess_text(input_text)
            assert "http" not in result
            assert "https" not in result

    def test_preprocess_text_removes_emails(self, processor):
        """Test that email addresses are removed"""
        test_cases = [
            ("Contact me at test@example.com", "Contact me at"),
            ("Email alice@test.org and bob@company.net", "Email and"),
            ("No emails here", "No emails here")
        ]
        
        for input_text, expected in test_cases:
            result = processor.preprocess_text(input_text)
            assert "@" not in result

    def test_preprocess_text_handles_special_characters(self, processor):
        """Test handling of special characters"""
        test_cases = [
            ("Normal punctuation!", "Normal punctuation!"),
            ("Question?", "Question?"),
            ("Hyphen-word", "Hyphen-word"),
            ("Quote 'text' here", "Quote 'text' here"),
            ("Weird ñoña çharacters", "Weird characters"),
        ]
        
        for input_text, expected in test_cases:
            result = processor.preprocess_text(input_text)
            # Basic punctuation should be preserved
            if any(char in input_text for char in "!?.'-"):
                assert any(char in result for char in "!?.'-") or result == ""

    def test_preprocess_text_handles_edge_cases(self, processor):
        """Test edge cases for text preprocessing"""
        assert processor.preprocess_text("") == ""
        assert processor.preprocess_text(None) == ""
        assert processor.preprocess_text("   ") == ""
        assert processor.preprocess_text("Multiple    spaces") == "Multiple spaces"

    def test_create_conversations_groups_by_time(self, processor, sample_messages):
        """Test that messages are grouped into conversations by time gaps"""
        conversations = processor.create_conversations(sample_messages)
        
        # Should have 1 conversation (the last single message doesn't meet min_length requirement)
        assert len(conversations) == 1
        assert len(conversations[0]) == 3  # First 3 messages that had content after preprocessing

    def test_create_conversations_filters_by_min_length(self, processor):
        """Test that conversations shorter than min_length are filtered out"""
        short_messages = [
            {"sender_name": "Alice", "timestamp_ms": 1000, "content": "Hi"}
        ]
        
        conversations = processor.create_conversations(short_messages)
        assert len(conversations) == 0  # No conversations meet min_length of 2

    def test_create_conversations_handles_empty_content(self, processor):
        """Test that messages without content are filtered out"""
        messages_with_empty = [
            {"sender_name": "Alice", "timestamp_ms": 1000, "content": "Hi"},
            {"sender_name": "Bob", "timestamp_ms": 1001, "content": ""},
            {"sender_name": "Alice", "timestamp_ms": 1002, "content": "How are you?"}
        ]
        
        conversations = processor.create_conversations(messages_with_empty)
        assert len(conversations) == 1
        assert len(conversations[0]) == 2  # Empty content message filtered out

    def test_create_training_pairs_general(self, processor):
        """Test creating training pairs without name filtering"""
        sample_conv = [
            {"sender_name": "Alice", "content": "Hello"},
            {"sender_name": "Bob", "content": "Hi there"},
            {"sender_name": "Alice", "content": "How are you?"}
        ]
        
        training_data = processor.create_training_pairs([sample_conv])
        
        assert len(training_data) == 2  # 2 pairs from 3 messages
        assert training_data[0]["prompt"] == "Hello"
        assert training_data[0]["response"] == "Hi there"
        assert training_data[1]["prompt"] == "Hi there"
        assert training_data[1]["response"] == "How are you?"

    def test_create_training_pairs_with_name_filter(self, processor):
        """Test creating training pairs with name filtering"""
        sample_conv = [
            {"sender_name": "Alice", "content": "Hello"},
            {"sender_name": "Bob", "content": "Hi there"},
            {"sender_name": "Alice", "content": "How are you?"},
            {"sender_name": "Bob", "content": "I'm good"}
        ]
        
        training_data = processor.create_training_pairs([sample_conv], your_name="Bob")
        
        assert len(training_data) == 2  # Only pairs where Bob responds
        assert training_data[0]["response"] == "Hi there"
        assert training_data[1]["response"] == "I'm good"

    def test_get_context(self, processor):
        """Test context extraction from conversations"""
        conversation = [
            {"sender_name": "Alice", "content": "Message 1"},
            {"sender_name": "Bob", "content": "Message 2"},
            {"sender_name": "Alice", "content": "Message 3"},
            {"sender_name": "Bob", "content": "Message 4"}
        ]
        
        context = processor._get_context(conversation, 3, context_window=2)
        expected_lines = ["Bob: Message 2", "Alice: Message 3"]
        
        for line in expected_lines:
            assert line in context

    @pytest.fixture
    def temp_json_file(self, sample_json_data):
        """Create a temporary JSON file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)

    def test_load_messenger_data_single_file(self, temp_json_file):
        """Test loading data from a single JSON file"""
        processor = MessengerDataProcessor(temp_json_file)
        messages = processor.load_messenger_data()
        
        assert len(messages) == 4
        assert messages[0]["sender_name"] == "Alice"

    def test_load_messenger_data_directory(self, sample_json_data):
        """Test loading data from a directory with multiple JSON files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple JSON files
            file1_path = Path(temp_dir) / "file1.json"
            file2_path = Path(temp_dir) / "file2.json"
            
            with open(file1_path, 'w') as f:
                json.dump({"messages": sample_json_data["messages"][:2]}, f)
            
            with open(file2_path, 'w') as f:
                json.dump({"messages": sample_json_data["messages"][2:]}, f)
            
            processor = MessengerDataProcessor(temp_dir)
            messages = processor.load_messenger_data()
            
            assert len(messages) == 4

    def test_prepare_dataset_integration(self, temp_json_file):
        """Test the complete dataset preparation pipeline"""
        processor = MessengerDataProcessor(temp_json_file, min_conversation_length=2)
        
        with patch.object(processor, 'preprocess_text') as mock_preprocess:
            # Mock preprocessing to return cleaned text
            mock_preprocess.side_effect = lambda x: x.replace("😀", "").replace("🎉", "").strip() if x else ""
            
            df = processor.prepare_dataset()
            
            assert len(df) > 0
            assert "prompt" in df.columns
            assert "response" in df.columns
            assert "context" in df.columns
            assert "total_length" in df.columns

    def test_prepare_dataset_filters_long_conversations(self, temp_json_file):
        """Test that very long conversations are filtered out"""
        processor = MessengerDataProcessor(temp_json_file, max_context_length=10)  # Very small limit
        
        df = processor.prepare_dataset()
        
        # All conversations should be under the length limit
        assert all(df["total_length"] < 10)


class TestMacOSLLMFineTuner:
    """Test suite for MacOSLLMFineTuner class"""

    @pytest.fixture
    def fine_tuner(self):
        """Create a MacOSLLMFineTuner instance for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield MacOSLLMFineTuner(
                base_model="test-model",
                output_dir=temp_dir,
                use_mps=False  # Force CPU for testing
            )

    def test_detect_device_cpu_fallback(self, fine_tuner):
        """Test device detection falls back to CPU"""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            
            device = fine_tuner._detect_device()
            assert device == "cpu"

    def test_detect_device_mps_available(self, fine_tuner):
        """Test MPS device detection when available"""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True), \
             patch('torch.zeros') as mock_zeros:
            
            # Mock successful MPS tensor creation
            mock_tensor = MagicMock()
            mock_zeros.return_value.to.return_value = mock_tensor
            
            device = fine_tuner._detect_device()
            assert device == "mps"

    def test_detect_device_mps_not_functional(self, fine_tuner):
        """Test MPS detection when available but not functional"""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True), \
             patch('torch.zeros') as mock_zeros:
            
            # Mock MPS tensor creation failure
            mock_zeros.return_value.to.side_effect = Exception("MPS not functional")
            
            device = fine_tuner._detect_device()
            assert device == "cpu"

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing"""
        import pandas as pd
        return pd.DataFrame([
            {"prompt": "Hello", "response": "Hi there", "context": ""},
            {"prompt": "How are you?", "response": "I'm good", "context": "Previous: Hello"}
        ])

    def test_format_dataset_structure(self, fine_tuner, sample_dataframe):
        """Test dataset formatting creates correct structure"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }
        
        with patch('finetuning.Dataset.from_list') as mock_dataset, \
             patch.object(mock_dataset.return_value, 'map') as mock_map:
            
            mock_dataset.return_value.column_names = ["text"]
            mock_map.return_value = MagicMock()
            
            result = fine_tuner.format_dataset(sample_dataframe, mock_tokenizer)
            
            # Verify Dataset.from_list was called with correct format
            mock_dataset.assert_called_once()
            call_args = mock_dataset.call_args[0][0]
            
            assert len(call_args) == 2  # Two rows
            assert "text" in call_args[0]
            assert "User:" in call_args[0]["text"]
            assert "Assistant:" in call_args[0]["text"]


class TestIntegration:
    """Integration tests for the complete pipeline"""

    def test_end_to_end_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline"""
        # Create sample data with various preprocessing challenges
        sample_data = {
            "messages": [
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1000000,
                    "content": "Hello! 😀 Check https://example.com"
                },
                {
                    "sender_name": "Bob",
                    "timestamp_ms": 1000001,
                    "content": "Got it! Email me at bob@test.com 🎉"
                },
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1000002,
                    "content": "Will do! ñoña special chars here"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_path = f.name
        
        try:
            processor = MessengerDataProcessor(temp_path)
            df = processor.prepare_dataset()
            
            # Verify preprocessing worked
            all_text = " ".join(df["prompt"].tolist() + df["response"].tolist())
            
            # Should not contain emojis, URLs, or emails
            assert "😀" not in all_text
            assert "🎉" not in all_text
            assert "https://" not in all_text
            assert "@" not in all_text
            
            # Should contain cleaned text
            assert len(df) > 0
            assert all(df["total_length"] > 0)
            
        finally:
            os.unlink(temp_path)

    def test_conversation_splitting_and_preprocessing(self):
        """Test that conversation splitting works with preprocessing"""
        # Messages with long time gap to test conversation splitting
        sample_data = {
            "messages": [
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1000000,
                    "content": "First conversation 😀"
                },
                {
                    "sender_name": "Bob", 
                    "timestamp_ms": 1000001,
                    "content": "Reply in first 🎉"
                },
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 3000000,  # 33+ minutes later
                    "content": "New conversation https://test.com"
                },
                {
                    "sender_name": "Bob",
                    "timestamp_ms": 3000001,
                    "content": "Reply in second bob@test.com"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_path = f.name
        
        try:
            processor = MessengerDataProcessor(temp_path)
            messages = processor.load_messenger_data()
            conversations = processor.create_conversations(messages)
            
            # Should create 2 separate conversations
            assert len(conversations) == 2
            
            # Verify preprocessing was applied
            for conv in conversations:
                for msg in conv:
                    content = msg["content"]
                    assert "😀" not in content
                    assert "🎉" not in content
                    assert "https://" not in content
                    assert "@" not in content
                    
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])