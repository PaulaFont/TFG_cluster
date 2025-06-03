import re, time, os, json
from collections import Counter
from voting_utils import _detokenize

# --- Configuration ---
GAP_TOKEN = "<GAP>" # Represents a missing word in an alignment/column

# --- Tokenization and Detokenization ---
def tokenize_into_words(text):
    """
    Tokenizes text into words and punctuation.
    - Converts to lowercase.
    - Separates punctuation from words.
    """
    if not isinstance(text, str):
        text = str(text)
    #text = text.lower() #TODO: I commented this
    # Keep words (alphanumeric + apostrophes/hyphens within words) and punctuation separate
    # This regex tries to capture common word patterns and individual punctuation marks
    tokens = re.findall(r"[\w'-]+|[^\w\s]", text)
    return [t for t in tokens if t.strip()] # Remove any empty strings from regex artifacts

def detokenize_words(tokens):
    """
    Detokenizes a list of word/punctuation tokens back into a string
    with basic smart spacing.
    """
    if not tokens:
        return ""
    
    text = ""
    for i, token in enumerate(tokens):
        if i == 0:
            text = token
        else:
            # Basic heuristic for spacing:
            # No space before common punctuation unless it's an opening bracket/quote
            # No space after opening brackets/quotes
            prev_token = tokens[i-1]
            if token in ".,;:!?)'" and prev_token not in "([\"'": # Closing punctuation
                text += token
            elif prev_token in "([\"'" and token not in ")]\"'": # Content after opening
                 text += token
            elif token == "'" and prev_token.endswith("s"): # Possessive like students'
                text += token
            else: # Default: add a space
                text += " " + token
    
    # Minor cleanups (can be expanded)
    text = text.replace(" 's", "'s") # for cases like "it 's" -> "it's"
    return text

# --- Core Word-Level Harmonization (Direct Word-Column Voting) ---
def harmonize_llm_outputs_word_level(llm_outputs):
    """
    Harmonizes multiple LLM outputs using direct word-column voting.
    Assumes structural similarity between outputs.
    """
    if not llm_outputs:
        return ""
    
    # Ensure all inputs are strings
    processed_outputs = [str(out) if out is not None else "" for out in llm_outputs]

    if len(processed_outputs) == 1:
        return processed_outputs[0] # No harmonization needed for a single output

    # 1. Tokenize each LLM output into a list of words
    tokenized_outputs = [tokenize_into_words(doc) for doc in processed_outputs]

    # 2. Determine the maximum number of words in any output
    max_word_count = 0
    if any(tokenized_outputs):
        non_empty_token_lists = [tl for tl in tokenized_outputs if tl]
        if non_empty_token_lists:
            max_word_count = max(len(tokens) for tokens in non_empty_token_lists)

    if max_word_count == 0: # All outputs were empty or resulted in no tokens
        return ""

    # 3. Iterate word by word (column by column) and vote
    final_voted_words = []
    for word_idx in range(max_word_count):
        current_word_column = []
        for token_list in tokenized_outputs:
            if word_idx < len(token_list):
                current_word_column.append(token_list[word_idx])
            else:
                current_word_column.append(GAP_TOKEN) # This output is shorter

        # Vote on the words in the current_word_column
        if not current_word_column: # Should not happen
            final_voted_words.append(GAP_TOKEN) # Or skip
            continue

        word_counts = Counter(current_word_column)
        
        # Simple majority vote
        # Tie-breaking:
        # 1. Prefer non-GAP_TOKEN if tied with GAP_TOKEN.
        # 2. If multiple non-GAP_TOKENs tie, prefer the one that appears in more original (non-padded) contexts,
        #    or alphabetically, or based on input order. For now, let's do alphabetical.
        most_common_list = word_counts.most_common()
        
        if not most_common_list:
            # This case implies current_word_column was empty, handled above
            chosen_word = GAP_TOKEN 
        else:
            winner_word = most_common_list[0][0]
            max_count = most_common_list[0][1]

            if winner_word == GAP_TOKEN and len(most_common_list) > 1:
                # Check if any actual word has the same max_count
                tied_actual_words = [
                    word for word, count in most_common_list
                    if word != GAP_TOKEN and count == max_count
                ]
                if tied_actual_words:
                    # Tune: Tie-breaking for actual words. Alphabetical is simple.
                    # Could also consider original frequency or average position.
                    winner_word = min(tied_actual_words) 
            
            chosen_word = winner_word
        
        # Only add the word if it's not a GAP that won overwhelmingly
        # or if we want to preserve gaps for structural reasons (for now, let's keep non-gap winners)
        if chosen_word != GAP_TOKEN:
            final_voted_words.append(chosen_word)
        # Tune: How to handle GAP_TOKENs that "win" a vote or are the only option.
        # If most LLMs agree there's "nothing" at a word position (relative to the longest LLM output),
        # we might want to represent that by not adding anything to final_voted_words for that column.
        # Current logic: only appends non-GAP winners. This means the output length
        # will be closer to the consensus length, not necessarily max_word_count.

    # 4. Detokenize the final list of voted words
    harmonized_text = detokenize_words(final_voted_words)
    
    return harmonized_text

# --- Main Example ---
if __name__ == "__main__":
    base_directory = "/data/users/pfont/"
    json_path = os.path.join(base_directory , "llm_versions.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
    llm_outputs_example = data_dict["278"]["llm_version_text"]

    print("\n--- Harmonizing LLM Outputs (Word-Level Direct Column Voting) ---")
    harmonized_result = harmonize_llm_outputs_word_level(llm_outputs_example)

    print("\n--- Final Harmonized Text ---")
    print(harmonized_result)