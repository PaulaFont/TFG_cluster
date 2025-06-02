import re
from collections import Counter
import os

def _tokenize(text, mode='word'):
    """Tokenizes text either by word or by character."""
    if mode == 'word':
        # Normalize: lowercase and unify whitespace before word tokenization
        processed_text = text.lower()
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        tokens = re.findall(r"[\w'-]+|[^\w\s]", processed_text) # Keeps words and punctuation separate
        return [t for t in tokens if t.strip()]
    elif mode == 'char':
        return list(text) # Each character is a token
    else:
        raise ValueError("Mode must be 'word' or 'char'")

def _detokenize(tokens, mode='word'):
    """Detokenizes a list of tokens back into a string."""
    if not tokens:
        return ""
    if mode == 'word':
        text = tokens[0]
        for i in range(1, len(tokens)):
            current_token = tokens[i]
            previous_token = tokens[i-1]
            # Basic rules for punctuation (can be improved for more complex cases)
            if current_token in ".,;:!?)'" and not (current_token == "'" and previous_token.isalnum()):
                text += current_token
            elif previous_token in "([¿¡\"'" and not current_token in ")]\"'": # No space after opening
                 text += current_token
            else:
                text += " " + current_token
        # Clean up common detokenization artifacts
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
        text = text.replace(" :", ":").replace(" ;", ";")
        text = text.replace("( ", "(").replace(" )", ")")
        text = text.replace("[ ", "[").replace(" ]", "]")
        return text
    elif mode == 'char':
        return "".join(tokens)
    else:
        raise ValueError("Mode must be 'word' or 'char'")

def normalize_text(text):

    """Basic normalization: lowercase and unify whitespace."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text
