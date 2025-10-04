# Toy Grammar Example for Text Reformatting

## Input Text
"The cat sat"

## Grammar Complexity Analysis

### States Needed
For 3 words and 2 formatting types (bold, italic):
- Each position needs 2^2 = 4 states (plain, bold, italic, bold+italic)
- Total states = 3 words × 4 states = 12 states

### Generated Grammar

```
root ::= pos-0-plain

# Position 0: "The"
pos-0-plain ::= "The" " " pos-1-plain
              | "**" "The" " " pos-1-bold
              | "*" "The" " " pos-1-italic

pos-0-bold ::= "The" " " pos-1-bold
             | "The" "**" " " pos-1-plain
             | "*" "The" " " pos-1-bold-italic

pos-0-italic ::= "The" " " pos-1-italic
               | "The" "*" " " pos-1-plain
               | "**" "The" " " pos-1-bold-italic

pos-0-bold-italic ::= "The" " " pos-1-bold-italic
                    | "The" "**" " " pos-1-italic
                    | "The" "*" " " pos-1-bold

# Position 1: "cat"
pos-1-plain ::= "cat" " " pos-2-plain
              | "**" "cat" " " pos-2-bold
              | "*" "cat" " " pos-2-italic

pos-1-bold ::= "cat" " " pos-2-bold
             | "cat" "**" " " pos-2-plain
             | "*" "cat" " " pos-2-bold-italic

pos-1-italic ::= "cat" " " pos-2-italic
               | "cat" "*" " " pos-2-plain
               | "**" "cat" " " pos-2-bold-italic

pos-1-bold-italic ::= "cat" " " pos-2-bold-italic
                    | "cat" "**" " " pos-2-italic
                    | "cat" "*" " " pos-2-bold

# Position 2: "sat" (final)
pos-2-plain ::= "sat"

pos-2-bold ::= "sat" "**"
             | "*" "sat" "*" "**"

pos-2-italic ::= "sat" "*"
               | "**" "sat" "**" "*"

pos-2-bold-italic ::= "sat" "**" "*"
                    | "sat" "*" "**"
```

## Possible Outputs
- Plain: `The cat sat`
- Bold all: `**The cat sat**`
- Bold "cat": `The **cat** sat`
- Italic "The", bold "sat": `*The* cat **sat**`
- Nested: `**The *cat* sat**`

## Complexity Confirmation
- Words (n) = 3
- Formatting types (f) = 2 (bold, italic)
- States per position = 2^f = 4
- Total states = n × 2^f = 3 × 4 = 12

**You're right!** It's O(n × 2^f) which is linear in n.
For typical f=2 or f=3, this is just O(n × 4) or O(n × 8).
Totally manageable!