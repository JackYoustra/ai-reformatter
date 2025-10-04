"""
Grammar generator for text reformatting with exact word preservation.
Generates a context-free grammar that enforces exact word ordering
while allowing the LLM to choose formatting placement.
"""

import re
from typing import List, Set, Dict
from enum import Enum


class FormatState(Enum):
    """Possible formatting states"""
    PLAIN = "plain"
    BOLD = "bold"
    ITALIC = "italic"
    BOLD_ITALIC = "bold-italic"


def escape_for_grammar(word: str) -> str:
    """Escape special characters for grammar rules"""
    # Escape quotes and backslashes
    word = word.replace('\\', '\\\\')
    word = word.replace('"', '\\"')
    return word


def tokenize_text(text: str) -> List[str]:
    """Split text into words while preserving punctuation"""
    # Simple tokenization - can be improved
    # Split on whitespace but keep punctuation attached to words
    tokens = text.split()
    return tokens


def generate_grammar(text: str, allow_paragraphs: bool = True) -> str:
    """
    Generate a CFG that enforces exact word sequence with formatting options.

    Args:
        text: Input text to preserve exactly
        allow_paragraphs: Whether to allow paragraph breaks

    Returns:
        Grammar string in extended BNF format
    """
    tokens = tokenize_text(text)
    if not tokens:
        return 'root ::= ""'

    grammar_lines = []

    # Add root rule
    grammar_lines.append('root ::= seq-0-plain')
    grammar_lines.append('')

    # Define formatting markers
    grammar_lines.append('# Formatting markers')
    grammar_lines.append('bold-open ::= "**"')
    grammar_lines.append('bold-close ::= "**"')
    grammar_lines.append('italic-open ::= "*"')
    grammar_lines.append('italic-close ::= "*"')
    if allow_paragraphs:
        grammar_lines.append('para-break ::= "\\n\\n"')
    grammar_lines.append('')

    # Generate rules for each position and state
    for i, token in enumerate(tokens):
        escaped_token = escape_for_grammar(token)
        is_last = (i == len(tokens) - 1)
        next_pos = i + 1

        grammar_lines.append(f'# Position {i}: "{token}"')

        # Plain state
        if is_last:
            # Last token - must close any formatting
            grammar_lines.append(f'seq-{i}-plain ::= "{escaped_token}"')
        else:
            rules = []
            # Can stay plain
            rules.append(f'"{escaped_token}" " " seq-{next_pos}-plain')
            # Can open bold
            rules.append(f'bold-open "{escaped_token}" " " seq-{next_pos}-bold')
            # Can open italic
            rules.append(f'italic-open "{escaped_token}" " " seq-{next_pos}-italic')
            # Can add paragraph break (if not last)
            if allow_paragraphs:
                rules.append(f'"{escaped_token}" para-break seq-{next_pos}-plain')

            grammar_lines.append(f'seq-{i}-plain ::= {" | ".join(rules)}')

        # Bold state
        if is_last:
            # Must close bold
            grammar_lines.append(f'seq-{i}-bold ::= "{escaped_token}" bold-close')
        else:
            rules = []
            # Can stay bold
            rules.append(f'"{escaped_token}" " " seq-{next_pos}-bold')
            # Can close bold
            rules.append(f'"{escaped_token}" bold-close " " seq-{next_pos}-plain')
            # Can open italic (nested)
            rules.append(f'italic-open "{escaped_token}" " " seq-{next_pos}-bold-italic')
            # Can close and reopen for paragraph
            if allow_paragraphs:
                rules.append(f'"{escaped_token}" bold-close para-break seq-{next_pos}-plain')
                rules.append(f'"{escaped_token}" bold-close para-break bold-open seq-{next_pos}-bold')

            grammar_lines.append(f'seq-{i}-bold ::= {" | ".join(rules)}')

        # Italic state
        if is_last:
            # Must close italic
            grammar_lines.append(f'seq-{i}-italic ::= "{escaped_token}" italic-close')
        else:
            rules = []
            # Can stay italic
            rules.append(f'"{escaped_token}" " " seq-{next_pos}-italic')
            # Can close italic
            rules.append(f'"{escaped_token}" italic-close " " seq-{next_pos}-plain')
            # Can open bold (nested)
            rules.append(f'bold-open "{escaped_token}" " " seq-{next_pos}-bold-italic')
            # Can close and reopen for paragraph
            if allow_paragraphs:
                rules.append(f'"{escaped_token}" italic-close para-break seq-{next_pos}-plain')
                rules.append(f'"{escaped_token}" italic-close para-break italic-open seq-{next_pos}-italic')

            grammar_lines.append(f'seq-{i}-italic ::= {" | ".join(rules)}')

        # Bold-italic state
        if is_last:
            # Must close both - order matters for proper nesting
            grammar_lines.append(f'seq-{i}-bold-italic ::= "{escaped_token}" italic-close bold-close | "{escaped_token}" bold-close italic-close')
        else:
            rules = []
            # Can stay bold-italic
            rules.append(f'"{escaped_token}" " " seq-{next_pos}-bold-italic')
            # Can close italic only
            rules.append(f'"{escaped_token}" italic-close " " seq-{next_pos}-bold')
            # Can close bold only
            rules.append(f'"{escaped_token}" bold-close " " seq-{next_pos}-italic')
            # Can close both
            rules.append(f'"{escaped_token}" italic-close bold-close " " seq-{next_pos}-plain')
            rules.append(f'"{escaped_token}" bold-close italic-close " " seq-{next_pos}-plain')
            # Paragraph breaks require closing formatting
            if allow_paragraphs:
                rules.append(f'"{escaped_token}" italic-close bold-close para-break seq-{next_pos}-plain')
                rules.append(f'"{escaped_token}" bold-close italic-close para-break seq-{next_pos}-plain')

            grammar_lines.append(f'seq-{i}-bold-italic ::= {" | ".join(rules)}')

        grammar_lines.append('')

    return '\n'.join(grammar_lines)


def validate_grammar(grammar: str) -> bool:
    """Basic validation of generated grammar"""
    # Check for root rule
    if 'root ::=' not in grammar:
        return False

    # Check for balanced quotes
    quote_count = grammar.count('"')
    if quote_count % 2 != 0:
        return False

    # Check that all referenced rules are defined
    # This is a simplified check
    lines = grammar.split('\n')
    defined_rules = set()
    referenced_rules = set()

    for line in lines:
        if '::=' in line:
            rule_name = line.split('::=')[0].strip()
            defined_rules.add(rule_name)
            # Find referenced rules on RHS
            rhs = line.split('::=')[1]
            # Simple pattern matching for rule references
            words = rhs.split()
            for word in words:
                if word.startswith('seq-') or word in ['bold-open', 'bold-close',
                                                        'italic-open', 'italic-close',
                                                        'para-break', 'root']:
                    referenced_rules.add(word)

    # Check all referenced rules are defined
    undefined = referenced_rules - defined_rules
    if undefined and undefined != {'para-break'}:  # para-break is optional
        print(f"Undefined rules: {undefined}")
        return False

    return True


if __name__ == "__main__":
    # Test with a simple example
    test_text = "The cat sat on the mat"
    grammar = generate_grammar(test_text)
    print("Generated Grammar:")
    print("=" * 50)
    print(grammar)
    print("=" * 50)
    print(f"Grammar valid: {validate_grammar(grammar)}")

    # Show some stats
    lines = grammar.split('\n')
    rule_count = sum(1 for line in lines if '::=' in line)
    print(f"Total rules: {rule_count}")
    print(f"Words in input: {len(test_text.split())}")