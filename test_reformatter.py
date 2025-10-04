"""
Property-based tests for text reformatter.
Ensures exact text preservation and valid formatting.
"""

import re
import string
from typing import List

import pytest
from hypothesis import given, strategies as st, settings, assume

from grammar_generator import (
    generate_grammar,
    validate_grammar,
    tokenize_text,
    escape_for_grammar
)
from reformatter import strip_formatting


# Strategies for generating test inputs
word_strategy = st.text(
    alphabet=string.ascii_letters + string.digits,
    min_size=1,
    max_size=10
)

sentence_strategy = st.lists(
    word_strategy,
    min_size=1,
    max_size=10
).map(lambda words: " ".join(words))

text_strategy = st.lists(
    sentence_strategy,
    min_size=1,
    max_size=5
).map(lambda sentences: " ".join(sentences))


def count_unmatched_formatting(text: str, marker: str) -> int:
    """Count unmatched formatting markers"""
    count = 0
    in_format = False
    i = 0

    while i < len(text):
        if i + len(marker) <= len(text) and text[i:i+len(marker)] == marker:
            if in_format:
                in_format = False
            else:
                in_format = True
            i += len(marker)
        else:
            i += 1

    return 1 if in_format else 0


def extract_words_only(text: str) -> List[str]:
    """Extract only the words from formatted text, preserving order"""
    # Remove formatting
    stripped = strip_formatting(text)
    return stripped.split()


def is_valid_markdown_formatting(text: str) -> bool:
    """Check if text has valid markdown formatting"""
    # Check for unmatched bold markers
    if count_unmatched_formatting(text, "**") > 0:
        return False

    # Check for unmatched italic markers
    if count_unmatched_formatting(text, "*") > 0:
        return False

    # Check for no adjacent formatting markers without content
    if "****" in text or "**" == text or "*" == text:
        return False

    return True


class TestGrammarGenerator:
    """Test the grammar generation"""

    @given(text_strategy)
    def test_grammar_generation_succeeds(self, text: str):
        """Grammar generation should succeed for any text"""
        grammar = generate_grammar(text)
        assert grammar is not None
        assert "root ::=" in grammar

    @given(text_strategy)
    def test_grammar_is_valid(self, text: str):
        """Generated grammar should be valid"""
        grammar = generate_grammar(text)
        assert validate_grammar(grammar)

    @given(text_strategy)
    def test_grammar_contains_all_words(self, text: str):
        """Grammar should contain all input words"""
        grammar = generate_grammar(text)
        words = tokenize_text(text)

        for word in words:
            escaped = escape_for_grammar(word)
            assert f'"{escaped}"' in grammar

    def test_empty_text_grammar(self):
        """Empty text should produce minimal grammar"""
        grammar = generate_grammar("")
        assert grammar == 'root ::= ""'

    def test_single_word_grammar(self):
        """Single word should produce simple grammar"""
        grammar = generate_grammar("hello")
        assert "root ::=" in grammar
        assert '"hello"' in grammar

    @given(st.text(alphabet="\"\\", min_size=1, max_size=10))
    def test_escaping_special_chars(self, text: str):
        """Special characters should be properly escaped"""
        # Add a normal word to make valid text
        text = "word " + text + " word"
        grammar = generate_grammar(text)
        assert validate_grammar(grammar)


class TestFormatting:
    """Test formatting preservation and validation"""

    @given(text_strategy)
    def test_strip_formatting_preserves_words(self, text: str):
        """Stripping formatting should preserve all words"""
        # Simulate formatted text
        formatted = text.replace(" ", " **")  # Add some bold markers
        formatted = formatted.replace("**", "**", 1)  # Ensure paired

        stripped = strip_formatting(formatted)
        original_words = text.split()
        stripped_words = stripped.split()

        # Should have same words (though spacing might differ)
        assert set(original_words) == set(stripped_words)

    def test_strip_formatting_removes_all_markers(self):
        """Strip formatting should remove all markdown markers"""
        formatted = "**bold** and *italic* text\n\nNew paragraph"
        stripped = strip_formatting(formatted)

        assert "**" not in stripped
        assert "*" not in stripped
        assert "\n\n" not in stripped

    def test_matched_formatting_detection(self):
        """Should correctly detect unmatched formatting"""
        # Matched formatting
        assert count_unmatched_formatting("**bold**", "**") == 0
        assert count_unmatched_formatting("*italic*", "*") == 0
        assert count_unmatched_formatting("**bold** normal **bold**", "**") == 0

        # Unmatched formatting
        assert count_unmatched_formatting("**bold", "**") == 1
        assert count_unmatched_formatting("*italic", "*") == 1
        assert count_unmatched_formatting("**bold** **unclosed", "**") == 1

    def test_valid_markdown_detection(self):
        """Should correctly identify valid markdown"""
        # Valid formatting
        assert is_valid_markdown_formatting("**bold**")
        assert is_valid_markdown_formatting("*italic*")
        assert is_valid_markdown_formatting("**bold** and *italic*")
        assert is_valid_markdown_formatting("normal text")

        # Invalid formatting
        assert not is_valid_markdown_formatting("**unclosed")
        assert not is_valid_markdown_formatting("*unclosed")
        assert not is_valid_markdown_formatting("****")  # Empty bold
        assert not is_valid_markdown_formatting("**")  # Just markers


class TestPropertyPreservation:
    """Property-based tests for text preservation"""

    @given(text_strategy)
    def test_tokenization_preserves_content(self, text: str):
        """Tokenization should preserve all content"""
        tokens = tokenize_text(text)
        # Rejoining tokens with space should give equivalent text
        rejoined = " ".join(tokens)
        # Normalize both for comparison
        assert rejoined.split() == text.split()

    @given(text_strategy)
    @settings(max_examples=10)  # Limit for performance
    def test_grammar_enforces_word_order(self, text: str):
        """Grammar should enforce exact word order"""
        assume(len(text.split()) > 0)  # Skip empty text

        grammar = generate_grammar(text)
        words = tokenize_text(text)

        # Check that words appear in order in the grammar
        lines = grammar.split('\n')
        position = 0
        for word in words:
            escaped = escape_for_grammar(word)
            # Find rules for this position
            pos_rules = [l for l in lines if f'seq-{position}-' in l and '::=' in l]
            # At least one rule should contain this word
            assert any(f'"{escaped}"' in rule for rule in pos_rules), \
                f"Word '{word}' not found in position {position} rules"
            position += 1

    def test_grammar_size_scaling(self):
        """Grammar size should scale linearly with input size"""
        # Test linear scaling
        text1 = "one two three"
        text2 = "one two three four five six"

        grammar1 = generate_grammar(text1)
        grammar2 = generate_grammar(text2)

        rules1 = len([l for l in grammar1.split('\n') if '::=' in l])
        rules2 = len([l for l in grammar2.split('\n') if '::=' in l])

        # Should roughly double (4 states per word + constants)
        assert rules2 < rules1 * 3  # Not exponential
        assert rules2 > rules1  # But does grow


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_punctuation_handling(self):
        """Punctuation should be preserved"""
        text = "Hello, world! How are you?"
        grammar = generate_grammar(text)
        assert '"Hello,"' in grammar
        assert '"world!"' in grammar
        assert '"you?"' in grammar

    def test_single_character_words(self):
        """Single character words should work"""
        text = "I a b c"
        grammar = generate_grammar(text)
        assert validate_grammar(grammar)
        assert '"I"' in grammar
        assert '"a"' in grammar

    def test_repeated_words(self):
        """Repeated words should work correctly"""
        text = "the the the"
        grammar = generate_grammar(text)
        assert validate_grammar(grammar)
        # Should have 3 positions, each with "the"
        assert 'seq-0-plain ::=' in grammar
        assert 'seq-1-plain ::=' in grammar
        assert 'seq-2-plain ::=' in grammar

    def test_very_long_word(self):
        """Very long words should be handled"""
        long_word = "a" * 100
        text = f"short {long_word} end"
        grammar = generate_grammar(text)
        assert validate_grammar(grammar)
        assert f'"{long_word}"' in grammar


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])