# DISCLAIMER

100% vibe coded. Not a serious project. Intention is to use in conjuction with a blob of voice text.

# Text Reformatter with Grammar-Constrained Generation

A tool that uses Fireworks AI's grammar mode to reformat text while preserving the exact word sequence. The LLM has sole discretion over formatting choices (bold, italic, paragraph breaks) but cannot change, add, or remove any words.

## How It Works

1. **Parses input text** into tokens/words
2. **Generates a Context-Free Grammar (CFG)** that enforces exact word ordering
3. **Calls Fireworks AI** with the grammar constraint
4. **LLM chooses formatting** placement while being constrained to output exact words
5. **Validates output** to ensure text preservation

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Create `.env` file:
```bash
cp .env.example .env
# Edit .env and add your Fireworks API key
```

3. Get a Fireworks API key from https://fireworks.ai/

## Usage

### Basic Usage
```bash
# Format text from a file
uv run python reformatter.py sample_input.txt

# Format from stdin
echo "The changing world order..." | uv run python reformatter.py -

# Save to file
uv run python reformatter.py input.txt -o formatted.md
```

### Options
- `-o, --output`: Output file (default: stdout)
- `-m, --model`: Fireworks model to use
- `-t, --temperature`: Sampling temperature (0-1, default: 0.7)
- `--no-paragraphs`: Disable paragraph breaks
- `-v, --verbose`: Show debug information
- `--system-prompt`: Custom formatting instructions

## Testing

Run property-based tests:
```bash
uv run pytest test_reformatter.py -v
```

The tests verify:
- Exact word preservation
- Proper formatting pair matching
- Grammar generation correctness
- Edge case handling

## Example

Input:
```
The changing world order. The times ahead will be radically different...
```

Possible Output:
```
**The changing world order.**

The times ahead will be *radically different*...
```

## How the Grammar Works (in the strict balance mode)

For each word position and formatting state, the grammar generates rules that:
- Enforce the exact next word
- Allow opening new formatting (if valid for current state)
- Allow closing current formatting (if something is open)
- Track formatting stack to ensure proper nesting

Example for "The cat sat":
```
seq-0-plain ::= "The" " " seq-1-plain
              | "**" "The" " " seq-1-bold
              | "*" "The" " " seq-1-italic
```

In the non-strict mode, the grammar is much simpler:
```
root ::= <any punctuation, whitespace, formatting, etc.> <word> <any punctuation, whitespace, formatting, etc.>
```