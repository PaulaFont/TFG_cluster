from collections import Counter
import re
import json
import os
from utils import save_final_output 

# Using a placeholder character that is unlikely to appear in your documents
# The paper uses '_', but that might be in text. Let's use '^' or some other symbol.
PLACEHOLDER = '^'

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

def _find_longest_common_substring_in_all(strings, min_lcs_len=1):
    """
    Finds the longest common substring present in ALL strings.
    Tries to find it starting from the shortest string.
    Returns the LCS and a list of its starting indices in each string,
    or (None, None) if no common substring of at least min_lcs_len is found.
    """
    if not strings or not all(s for s in strings): # Ensure all strings are non-empty and list is not empty
        return None, None

    # Use the shortest string as the source for candidate substrings
    # (sorted to ensure deterministic behavior if multiple strings have same min length)
    sorted_strings = sorted(strings, key=len)
    shortest_str = sorted_strings[0]
    
    if not shortest_str or len(shortest_str) < min_lcs_len:
        return None, None

    for length in range(len(shortest_str), min_lcs_len - 1, -1):
        for i in range(len(shortest_str) - length + 1):
            substring_candidate = shortest_str[i : i + length]
            
            # Check if this candidate is in all other strings
            # And find its first occurrence index in each
            indices = [-1] * len(strings)
            is_common_to_all = True
            
            # First, find in the string it came from (shortest_str)
            # To handle multiple occurrences, we need to map shortest_str back to its original position
            # For simplicity, let's just check all strings including the shortest one uniformly.
            for k, s_to_check in enumerate(strings):
                try:
                    indices[k] = s_to_check.index(substring_candidate) # Find first occurrence
                except ValueError:
                    is_common_to_all = False
                    break
            
            if is_common_to_all:
                return substring_candidate, indices
                
    return None, None

def align_texts_recursively(texts, min_lcs_len=1):
    """
    Aligns a list of text strings recursively based on LCS, as described in the paper.
    """
    # Base Case 1: If 0 or 1 text, they are "aligned" or there's nothing to align.
    if len(texts) <= 1:
        return texts

    # Base Case 2: If any text is empty, or no LCS found, pad all to max length.
    # This check also implicitly handles if min_lcs_len is too high for any commonality.
    if any(not t for t in texts): # If any string is empty
        max_len = 0
        if texts: # Check if texts list itself is not empty
             max_len = max(len(t) for t in texts) if any(texts) else 0 # Max length of non-empty strings
        return [t.ljust(max_len, PLACEHOLDER) for t in texts]

    lcs, indices = _find_longest_common_substring_in_all(texts, min_lcs_len)

    if lcs is None: # No common substring found
        max_len = max(len(t) for t in texts)
        return [t.ljust(max_len, PLACEHOLDER) for t in texts]

    # Recursive step:
    aligned_parts = [""] * len(texts)
    
    # 1. Align parts to the LEFT of LCS
    left_segments = [texts[i][:indices[i]] for i in range(len(texts))]
    # Only recurse if there's actually content in any of the left segments
    if any(left_segments): 
        aligned_left_parts = align_texts_recursively(left_segments, min_lcs_len)
    else:
        aligned_left_parts = [""] * len(texts) # All left segments were empty

    # 2. Align parts to the RIGHT of LCS
    right_segments = [texts[i][indices[i] + len(lcs):] for i in range(len(texts))]
    # Only recurse if there's actually content in any of the right segments
    if any(right_segments):
        aligned_right_parts = align_texts_recursively(right_segments, min_lcs_len)
    else:
        aligned_right_parts = [""] * len(texts) # All right segments were empty
        
    # 3. Combine: aligned_left + LCS + aligned_right
    for i in range(len(texts)):
        aligned_parts[i] = aligned_left_parts[i] + lcs + aligned_right_parts[i]
            
    return aligned_parts

def get_voted_string_from_aligned(aligned_texts):
    """
    Performs character-wise majority voting on a list of aligned text strings.
    """
    if not aligned_texts or not aligned_texts[0]: # No texts or texts are empty
        return ""

    num_aligned_chars = len(aligned_texts[0])
    voted_chars = []

    for i in range(num_aligned_chars):
        column_chars = [text[i] for text in aligned_texts]
        
        # Count character occurrences in the current column
        char_counts = Counter(column_chars)
        
        # Determine the winner (most common character)
        # Tie-breaking:
        # 1. If PLACEHOLDER is tied with non-PLACEHOLDERS, prefer non-PLACEHOLDER.
        # 2. If multiple non-PLACEHOLDERS tie, pick the first one alphabetically (or by original engine order if available).
        
        most_common_list = char_counts.most_common() # List of (char, count)
        
        if not most_common_list: # Should not happen if column_chars is not empty
            voted_chars.append(PLACEHOLDER) # Or some default
            continue

        winner = most_common_list[0][0]
        max_count = most_common_list[0][1]

        # Tie-breaking: Prefer non-placeholder if tied
        if winner == PLACEHOLDER and len(most_common_list) > 1:
            # Check if any non-placeholder has the same max_count
            tied_non_placeholders = [
                char for char, count in most_common_list 
                if char != PLACEHOLDER and count == max_count
            ]
            if tied_non_placeholders:
                winner = min(tied_non_placeholders) # Pick alphabetically smallest
                                               # or from a preferred engine if you have that info

        voted_chars.append(winner)
            
    return "".join(voted_chars)

def harmonize_ocr_outputs(ocr_versions, min_lcs_len=1, debug=False):
    """
    Main function to take multiple OCR outputs, align them, and vote for a final version.
    """
    if not ocr_versions:
        return ""
    if len(ocr_versions) == 1:
        return ocr_versions[0]

    # Ensure it's a list of strings
    ocr_versions_list = [str(v) for v in ocr_versions]

    aligned_texts = align_texts_recursively(ocr_versions_list, min_lcs_len)
    
    if debug:
        print("--- Aligned Texts ---")
        for i, text in enumerate(aligned_texts):
            print(f"V{i+1}: '{text}'")
        print("---------------------")

    voted_text_raw = get_voted_string_from_aligned(aligned_texts)
    
    if debug:
        print(f"Voted (raw): '{voted_text_raw}'")

    # Post-process the voted string:
    # 1. Replace placeholders with a single space.
    # 2. Collapse multiple spaces into one.
    # 3. Strip leading/trailing spaces.
    final_text = voted_text_raw.replace(PLACEHOLDER, ' ')
    final_text = re.sub(r'\s+', ' ', final_text).strip()
    
    return final_text

# --- Example Usage ---
def example_usage():
    print("--- Example 1 (from paper, conceptual) ---")
    texts1 = [
        "Ana has pears",
        "Joana loves deer",
        "Diana sheds tears"
    ]
    # Expected (from paper's Figure 3, but my algorithm might differ slightly in placeholder placement):
    # A__ n a _ h a _ s _ _ p e a r s
    # J o a n a _ l o v e s _ d e e r _ _
    # D i a n a _ s h e d s _ t e a r s
    # Voted: A_ana _h_a_s _ _ _e_ars (This is a guess, depends on tie-breaking & exact alignment)
    result1 = harmonize_ocr_outputs(texts1, min_lcs_len=1, debug=True)
    print(f"Final Harmonized 1: '{result1}'\n")

    print("--- Example 2 (more similar texts) ---")
    texts2 = [
        "the quick brown fox jumps",
        "the quick brown foox jumps over", # error 'o', extra ' over'
        "the quick brown fox jumps",
        "a quick brown fox jumps",   # error 'a'
        "the quik brown fox jumps"   # error 'i'
    ]
    # Expected: "the quick brown fox jumps" (ideally, 'over' might get dropped or kept based on votes)
    result2 = harmonize_ocr_outputs(texts2, min_lcs_len=2, debug=True)
    print(f"Final Harmonized 2: '{result2}'\n")

    print("--- Example 3 (different lengths and errors) ---")
    texts3 = [
        "This is version one of the document.",
        "This is version two.", # Shorter, different word
        "This is version one of document.", # Missing "the"
        "This as version one of the document.", # "as" vs "is"
        "This is version one of the document" # Missing final period
    ]
    result3 = harmonize_ocr_outputs(texts3, min_lcs_len=3, debug=True)
    print(f"Final Harmonized 3: '{result3}'\n")

    print("--- Example 4 (simple identical) ---")
    texts4 = ["test", "test", "test"]
    result4 = harmonize_ocr_outputs(texts4, debug=True)
    print(f"Final Harmonized 4: '{result4}'\n")

    print("--- Example 5 (simple different) ---")
    texts5 = ["apple", "apply", "apricot"]
    result5 = harmonize_ocr_outputs(texts5, min_lcs_len=2, debug=True)
    print(f"Final Harmonized 5: '{result5}'\n")
    
    print("--- Example 6 (edge case - one empty string) ---")
    texts6 = ["hello world", "", "hello universe"]
    result6 = harmonize_ocr_outputs(texts6, min_lcs_len=2, debug=True)
    print(f"Final Harmonized 6: '{result6}'\n")

    print("--- Example 7 (edge case - all empty strings) ---")
    texts7 = ["", "", ""]
    result7 = harmonize_ocr_outputs(texts7, min_lcs_len=1, debug=True)
    print(f"Final Harmonized 7: '{result7}'\n")
    
    print("--- Example 8 (edge case - single string) ---")
    texts8 = ["single string input"]
    result8 = harmonize_ocr_outputs(texts8, debug=True)
    print(f"Final Harmonized 8: '{result8}'\n")

def test(filename, output_folder="/home/pfont/pipeline/voting_tests", current_dir = "/data/users/pfont"):
    # Open all versions of one document
    versions = []
    for folder in os.listdir(current_dir):
        folder_path = os.path.join(current_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("out_transcription_"):
            file_path = os.path.join(folder_path, filename)
            if os.path.exists(file_path):
                print(f"Found: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                versions.append(content)

    harmonized_text = harmonize_ocr_outputs(versions, min_lcs_len=2)
    save_final_output(harmonized_text, filename, output_folder)

if __name__ == "__main__":
    # example_usage()

    """## Experiment with first paragraph
    with open('./miscelanious/voting_examples.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    my_ocr_versions = list(data.values())

    final_coherent_text = harmonize_ocr_outputs(my_ocr_versions, min_lcs_len=2, debug=True)

    print("Final Text:", final_coherent_text)"""

    ## Experiment with a couple of documents
    document_filenames = ["rsc37_rsc176_278.txt", "rsc37_rsc176_364.txt"]
    for filename in document_filenames:
        test(filename)