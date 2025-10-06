#!/usr/bin/env python3
"""
Text reformatter using Fireworks AI with grammar-constrained generation.
Preserves exact word sequence while allowing LLM to add formatting.
"""

import os
import sys
import argparse
import difflib
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fireworks.client import Fireworks

from grammar_generator import generate_grammar, validate_grammar, tokenize_text


# Load environment variables
load_dotenv()


def strip_formatting(text: str) -> str:
    """Remove all markdown formatting to get plain text"""
    # Remove bold
    text = text.replace("**", "")
    # Remove italic
    text = text.replace("*", "")
    # Replace paragraph breaks with single space
    text = text.replace("\n\n", " ")
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def validate_output(original: str, formatted: str, strict_spacing: bool = False) -> tuple[bool, Optional[str]]:
    """
    Validate that formatted output preserves original text.

    Args:
        original: Original input text
        formatted: Formatted output text
        strict_spacing: If True, validate exact text preservation. If False, validate word sequence only.

    Returns:
        Tuple of (is_valid, diff_output)
        - is_valid: True if text is preserved correctly
        - diff_output: None if valid, otherwise a git-style diff showing differences
    """
    if strict_spacing:
        # Strict mode: validate exact text preservation
        # Strip formatting from output
        stripped = strip_formatting(formatted)

        # Normalize whitespace in original
        original_normalized = " ".join(original.split())

        if stripped != original_normalized:
            # Generate unified diff
            diff = difflib.unified_diff(
                original_normalized.splitlines(keepends=True),
                stripped.splitlines(keepends=True),
                fromfile='expected (original)',
                tofile='actual (stripped output)',
                lineterm=''
            )
            diff_text = ''.join(diff)

            # If diff is empty (single-line difference), show character-level diff
            if not diff_text:
                diff = difflib.ndiff([original_normalized], [stripped])
                diff_text = '\n'.join(diff)

            return False, diff_text

        return True, None
    else:
        # Flexible mode: validate word sequence preservation
        from grammar_generator import tokenize_words
        original_words = tokenize_words(original)
        formatted_words = tokenize_words(formatted)

        if original_words != formatted_words:
            # Generate word-level diff
            diff = difflib.unified_diff(
                [' '.join(original_words)],
                [' '.join(formatted_words)],
                fromfile='expected (original words)',
                tofile='actual (output words)',
                lineterm=''
            )
            diff_text = '\n'.join(diff)

            return False, diff_text

        return True, None


def reformat_text(
    text: str,
    model: str = "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    strict_spacing: bool = False,
    allow_paragraphs: bool = True,
    verbose: bool = False
) -> tuple[str, Optional[str]]:
    """
    Reformat text using grammar-constrained generation.

    Args:
        text: Input text to reformat
        model: Fireworks model to use
        temperature: Sampling temperature (0-1)
        system_prompt: Optional system prompt to guide formatting choices
        strict_spacing: If True, preserve exact spacing (default: False)
        allow_paragraphs: Whether to allow paragraph breaks (strict mode only)
        verbose: Print debug information

    Returns:
        Tuple of (formatted_text, validation_diff)
        - formatted_text: The formatted output
        - validation_diff: None if valid, otherwise a diff showing where text was not preserved
    """
    # Initialize Fireworks client
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY not found in environment variables")

    client = Fireworks(api_key=api_key)

    # Generate grammar
    if verbose:
        print(f"Generating grammar (strict_spacing={strict_spacing})...")
    grammar = generate_grammar(text, strict_spacing=strict_spacing, allow_paragraphs=allow_paragraphs)

    if verbose:
        print(f"Grammar size: {len(grammar)} characters, {len(grammar.split(chr(10)))} lines")

    # Validate grammar
    if not validate_grammar(grammar):
        raise ValueError("Generated grammar is invalid")

    # Prepare messages
    if system_prompt is None:
        if strict_spacing:
            system_prompt = """You are a text formatter. Your task is to add markdown formatting (bold with **, italic with *, and paragraph breaks with \\n\\n) to make the text more readable and emphasize important points.

Guidelines:
- Use **bold** for key concepts, important terms, and emphasis
- Use *italics* for subtle emphasis, quotes, or asides
- Add paragraph breaks (\\n\\n) at natural stopping points to improve readability
- The grammar constrains you to output the exact words in order with exact spacing - you only choose where to add formatting
- Be thoughtful about formatting - not everything needs emphasis"""
        else:
            system_prompt = """You are a text formatter and editor. Your task is to reformat text by adding punctuation, markdown formatting (bold with **, italic with *), and paragraph breaks (\\n\\n) to make it more readable and emphasize important points.

Guidelines:
- Add proper punctuation (periods, commas, semicolons, etc.) where needed
- Use **bold** for key concepts, important terms, and emphasis
- Use *italics* for subtle emphasis, quotes, or asides
- Add paragraph breaks (\\n\\n) at natural stopping points to improve readability
- The grammar constrains you to output the exact words in the exact order - you can add any punctuation, whitespace, or formatting you want
- Be thoughtful about formatting and punctuation - improve readability while maintaining the original meaning"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Format this text to make it more readable:\n\n{text}"}
    ]

    # Call Fireworks API with grammar constraint
    if verbose:
        print("Calling Fireworks AI...")

    try:
        # Fireworks requires stream=True for max_tokens > 5000
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "grammar",
                "grammar": grammar
            },
            temperature=temperature,
            max_tokens=131072,  # Llama 3.1 70B max context window
            stream=True
        )

        # Collect streamed response
        formatted_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                formatted_text += chunk.choices[0].delta.content

        # Validate output preserves original text (or word sequence in flexible mode)
        _, diff = validate_output(text, formatted_text, strict_spacing=strict_spacing)

        return formatted_text, diff

    except Exception as e:
        raise RuntimeError(f"API call failed: {e}") from e


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Reformat text with AI while preserving exact words"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        help="Input file path or '-' for stdin"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="accounts/fireworks/models/llama-v3p1-70b-instruct",
        help="Fireworks model to use"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0-1)"
    )
    parser.add_argument(
        "--strict-spacing",
        action="store_true",
        help="Preserve exact spacing (default: flexible punctuation/whitespace)"
    )
    parser.add_argument(
        "--no-paragraphs",
        action="store_true",
        help="Disable paragraph breaks (strict spacing mode only)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt for formatting guidance"
    )
    parser.add_argument(
        "--dump-grammar",
        action="store_true",
        help="Generate and dump the grammar, then exit without calling API"
    )

    args = parser.parse_args()

    # Read input text
    if args.input == "-" or args.input is None:
        if args.verbose:
            print("Reading from stdin...", file=sys.stderr)
        text = sys.stdin.read()
    else:
        if args.verbose:
            print(f"Reading from {args.input}...", file=sys.stderr)
        with open(args.input, "r") as f:
            text = f.read()

    # Strip existing formatting if present (for re-formatting)
    text = text.strip()

    if not text:
        print("Error: No input text provided", file=sys.stderr)
        sys.exit(1)

    # Handle --dump-grammar flag
    if args.dump_grammar:
        if args.verbose:
            print("Generating grammar...", file=sys.stderr)
        grammar = generate_grammar(text, strict_spacing=args.strict_spacing, allow_paragraphs=not args.no_paragraphs)
        print(grammar)
        sys.exit(0)

    # Reformat text
    try:
        formatted, validation_diff = reformat_text(
            text,
            model=args.model,
            temperature=args.temperature,
            system_prompt=args.system_prompt,
            strict_spacing=args.strict_spacing,
            allow_paragraphs=not args.no_paragraphs,
            verbose=args.verbose
        )

        # Write output (always, even if validation failed)
        if args.output:
            if args.verbose:
                print(f"Writing to {args.output}...", file=sys.stderr)
            with open(args.output, "w") as f:
                f.write(formatted)
        else:
            print(formatted)

        # Check validation result
        if validation_diff:
            print("\n" + "="*80, file=sys.stderr)
            print("VALIDATION FAILED: Output text does not match input!", file=sys.stderr)
            print("="*80, file=sys.stderr)
            print("\nDiff (- expected, + actual):", file=sys.stderr)
            print(validation_diff, file=sys.stderr)
            print("\n" + "="*80, file=sys.stderr)
            if args.output:
                print(f"Output saved to {args.output} despite validation failure", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()