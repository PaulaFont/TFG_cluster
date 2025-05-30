import re
from collections import Counter
import os

import re
from collections import Counter

PLACEHOLDER_TOKEN = "<||PAD||>" # A more unique placeholder

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


_memo_find_lcs_tokens = {} # Memoization dictionary for LCS finding

def _find_majority_lcs_tokens(list_of_token_lists, min_versions_match, min_lcs_len):
    """
    Finds the longest common contiguous sublist of tokens present in at least
    min_versions_match lists.
    """
    # Memoization key (needs to be hashable)
    # For very large lists, hashing might be too slow or complex.
    best_lcs_tokens = None
    best_lcs_locations = None # {version_idx: (start, end), ...}
    best_lcs_score = 0  # Prioritize length, then number of matches

    # Find a non-empty list to use as the source for candidate substrings
    source_list_idx = -1
    for idx, tl in enumerate(list_of_token_lists):
        if tl: # Check if list is not None and not empty
            source_list_idx = idx
            break
    
    if source_list_idx == -1: # All lists are empty or None
        # _memo_find_lcs_tokens[memo_key_tuple] = (None, None)
        return None, None

    source_tokens = list_of_token_lists[source_list_idx]

    # Iterate from longest possible substring down to min_lcs_len
    for length in range(len(source_tokens), min_lcs_len - 1, -1):
        for i in range(len(source_tokens) - length + 1):
            candidate_lcs = source_tokens[i : i + length]
            
            current_match_locations = {}
            num_matches = 0

            for v_idx, tokens in enumerate(list_of_token_lists):
                if not tokens: continue # Skip empty or None lists

                # Efficiently find first occurrence of candidate_lcs in tokens
                # This is a naive O(N*M) substring search for lists.
                # Can be optimized (e.g. KMP for lists, or string conversion if appropriate)
                # For now, keeping it simple and understandable.
                for k in range(len(tokens) - length + 1):
                    if tokens[k : k + length] == candidate_lcs:
                        current_match_locations[v_idx] = (k, k + length)
                        num_matches += 1
                        break # Found first occurrence in this version
            
            if num_matches >= min_versions_match:
                # Score: length first, then number of matches
                current_score = (length * 1000) + num_matches 
                if current_score > best_lcs_score:
                    best_lcs_score = current_score
                    best_lcs_tokens = candidate_lcs
                    best_lcs_locations = current_match_locations
        
        # If an LCS is found for the current (longest) length, take it (greedy for length)
        if best_lcs_tokens and len(best_lcs_tokens) == length:
            # _memo_find_lcs_tokens[memo_key_tuple] = (best_lcs_tokens, best_lcs_locations)
            return best_lcs_tokens, best_lcs_locations
            
    # _memo_find_lcs_tokens[memo_key_tuple] = (best_lcs_tokens, best_lcs_locations) # Could be (None, None)
    return best_lcs_tokens, best_lcs_locations

# Memoization dictionary for recursive alignment
_memo_align_recursively = {}

def _align_recursively_token_lists(list_of_token_lists, min_versions_match, min_lcs_len,
                                   placeholder_token, recursion_depth, max_recursion_depth):
    """
    Recursively aligns lists of tokens based on majority LCS.
    """
    # Memoization key
    # memo_key_tuple = tuple(tuple(tl) if tl else tuple() for tl in list_of_token_lists) # Handle None
    # if memo_key_tuple in _memo_align_recursively:
    #    return _memo_align_recursively[memo_key_tuple]

    if recursion_depth > max_recursion_depth:
        # print(f"Warning: Max recursion depth {max_recursion_depth} reached. Padding remaining.")
        max_len = 0
        for tl in list_of_token_lists:
            if tl: max_len = max(max_len, len(tl))
        
        padded_lists = []
        for tl in list_of_token_lists:
            current_list = tl if tl else [] # Handle None for token list
            padding_needed = max_len - len(current_list)
            padded_lists.append(current_list + [placeholder_token] * padding_needed)
        return padded_lists

    common_lcs_tokens, lcs_locations = _find_majority_lcs_tokens(
        list_of_token_lists, min_versions_match, min_lcs_len
    )

    # Base Case: No common substring found or list_of_token_lists is empty
    if common_lcs_tokens is None or not list_of_token_lists:
        max_len = 0
        active_lists = [tl for tl in list_of_token_lists if tl is not None] # Filter out None lists for max_len calc
        if not active_lists and not list_of_token_lists: # All lists are None or input is empty
             return [[] for _ in list_of_token_lists] if list_of_token_lists else []


        for tl in active_lists: # Use active_lists for max_len
            max_len = max(max_len, len(tl))
        
        padded_lists = []
        for tl in list_of_token_lists: # Iterate original to maintain structure
            current_list = tl if tl else [] # Handle None for token list
            padding_needed = max_len - len(current_list)
            padded_lists.append(current_list + [placeholder_token] * padding_needed)
        # _memo_align_recursively[memo_key_tuple] = padded_lists
        return padded_lists

    # Recursive Step
    num_input_lists = len(list_of_token_lists)
    aligned_outputs = [[] for _ in range(num_input_lists)]

    # 1. Align Left Parts
    left_parts_to_align = []
    for i in range(num_input_lists):
        token_list = list_of_token_lists[i]
        if token_list is None: # Handle None lists gracefully
            left_parts_to_align.append(None) # Propagate None
            continue
        if i in lcs_locations:
            start_idx, _ = lcs_locations[i]
            left_parts_to_align.append(token_list[:start_idx])
        else:
            left_parts_to_align.append(token_list) # Whole list is "left"

    aligned_left_sub_results = _align_recursively_token_lists(
        left_parts_to_align, min_versions_match, min_lcs_len,
        placeholder_token, recursion_depth + 1, max_recursion_depth
    )
    for i in range(num_input_lists):
        aligned_outputs[i].extend(aligned_left_sub_results[i])

    # 2. Add Common Substring or Placeholders
    len_common_lcs = len(common_lcs_tokens)
    for i in range(num_input_lists):
        if i in lcs_locations:
            aligned_outputs[i].extend(common_lcs_tokens)
        else:
            aligned_outputs[i].extend([placeholder_token] * len_common_lcs)

    # 3. Align Right Parts
    right_parts_to_align = []
    for i in range(num_input_lists):
        token_list = list_of_token_lists[i]
        if token_list is None: # Handle None lists
            right_parts_to_align.append(None)
            continue
        if i in lcs_locations:
            _, end_idx = lcs_locations[i]
            right_parts_to_align.append(token_list[end_idx:])
        else:
            right_parts_to_align.append([]) # No right part if anchor wasn't in this list

    aligned_right_sub_results = _align_recursively_token_lists(
        right_parts_to_align, min_versions_match, min_lcs_len,
        placeholder_token, recursion_depth + 1, max_recursion_depth
    )
    for i in range(num_input_lists):
        aligned_outputs[i].extend(aligned_right_sub_results[i])
    
    # _memo_align_recursively[memo_key_tuple] = aligned_outputs
    return aligned_outputs

def _perform_voting_on_aligned_tokens(aligned_versions_tokens, 
                                     placeholder_token_padding, # Renamed for clarity
                                     tie_breaker_version_idx,
                                     tie_resolution_strategy='prefer_version_idx',
                                     tie_placeholder_output="[AMBIGUOUS]"): # New placeholder for ties
    """Performs column-wise majority voting on aligned token lists."""
    if not aligned_versions_tokens or not aligned_versions_tokens[0]:
        return []

    final_harmonized_tokens = []
    num_aligned_positions = len(aligned_versions_tokens[0])
    num_versions = len(aligned_versions_tokens)

    for j in range(num_aligned_positions): # Iterate column by column
        column_tokens_for_vote = []
        original_indices_in_vote = [] 

        for i in range(num_versions):
            if j < len(aligned_versions_tokens[i]): 
                token = aligned_versions_tokens[i][j]
                if token != placeholder_token_padding: # Exclude padding placeholders from vote
                    column_tokens_for_vote.append(token)
                    original_indices_in_vote.append(i)
        
        if column_tokens_for_vote:
            counts = Counter(column_tokens_for_vote)
            top_tokens_with_counts = counts.most_common()
            
            if not top_tokens_with_counts: continue 

            # Check for a clear winner (not a tie for the top spot)
            is_tie_for_top = len(top_tokens_with_counts) > 1 and \
                             top_tokens_with_counts[0][1] == top_tokens_with_counts[1][1]

            if not is_tie_for_top: # Clear winner
                winner_token = top_tokens_with_counts[0][0]
                final_harmonized_tokens.append(winner_token)
            else: # Tie for the top spot
                if tie_resolution_strategy == 'prefer_version_idx':
                    tied_candidates = [tc[0] for tc in top_tokens_with_counts if tc[1] == top_tokens_with_counts[0][1]]
                    winner_token = top_tokens_with_counts[0][0] # Default winner before tie-breaking

                    found_preferred_winner = False
                    for k_idx, orig_v_idx in enumerate(original_indices_in_vote):
                        # Check if the token at this position in the original vote list
                        # belongs to the preferred tie_breaker_version_idx AND is one of the tied top candidates
                        if orig_v_idx == tie_breaker_version_idx and column_tokens_for_vote[k_idx] in tied_candidates:
                            winner_token = column_tokens_for_vote[k_idx]
                            found_preferred_winner = True
                            break
                    
                    if not found_preferred_winner:
                        min_original_idx_for_tie = float('inf')
                        # Iterate through tied candidates and find the one from the earliest original version
                        for tc_candidate in tied_candidates:
                            for k_idx, orig_v_idx in enumerate(original_indices_in_vote):
                                if column_tokens_for_vote[k_idx] == tc_candidate:
                                    if orig_v_idx < min_original_idx_for_tie:
                                        min_original_idx_for_tie = orig_v_idx
                                        winner_token = tc_candidate
                                    break # Found first occurrence of this tc_candidate
                    final_harmonized_tokens.append(winner_token)

                elif tie_resolution_strategy == 'omit':
                    pass # Do nothing, effectively omitting the token at this position
                
                elif tie_resolution_strategy == 'placeholder':
                    final_harmonized_tokens.append(tie_placeholder_output)
                
                else: # Default to preferring version index if strategy is unknown
                    # (This part is a fallback, ideally strategy is validated earlier)
                    winner_token = top_tokens_with_counts[0][0] # Fallback, could reuse 'prefer_version_idx' logic
                    final_harmonized_tokens.append(winner_token)
            
    return final_harmonized_tokens

def harmonize_documents(versions_text, 
                        mode='word', 
                        min_versions_match=None, 
                        min_lcs_len=3, 
                        tie_breaker_version_idx=0,
                        max_recursion_depth=50,
                        tie_resolution_strategy='prefer_version_idx', # New parameter
                        tie_placeholder_output="[AMBIGUOUS]" # New parameter
                        ):
    """
    Harmonizes multiple text versions using recursive LCS alignment and voting.

    Args:
        versions_text (list of str): A list of text strings.
        mode (str): 'word' for word-level or 'char' for character-level processing.
        min_versions_match (int, optional): Minimum number of versions that must share an LCS.
                                            Defaults to simple majority.
        min_lcs_len (int): Minimum length of a common substring to be considered an anchor.
        tie_breaker_version_idx (int): Index of the version to prefer in case of voting ties
                                       (used if tie_resolution_strategy is 'prefer_version_idx').
        max_recursion_depth (int): Maximum depth for the alignment recursion.
        tie_resolution_strategy (str): How to handle ties in voting.
                                       Options: 'prefer_version_idx', 'omit', 'placeholder'.
        tie_placeholder_output (str): The placeholder to use if strategy is 'placeholder' and a tie occurs.
                                      (For char mode, this should ideally be a single char like '_').

    Returns:
        str: The harmonized text.
    """
    if not versions_text:
        return ""
    
    num_versions = len(versions_text)
    if num_versions == 0: return ""

    if min_versions_match is None:
        min_versions_match = 2
    if min_versions_match < 1:
        min_versions_match = 1
    
    if mode == 'char' and tie_resolution_strategy == 'placeholder' and len(tie_placeholder_output) > 1:
        print(f"Warning: For 'char' mode with 'placeholder' tie resolution, "
              f"tie_placeholder_output ('{tie_placeholder_output}') should ideally be a single character.")


    _memo_find_lcs_tokens.clear()
    _memo_align_recursively.clear()

    initial_token_lists = [_tokenize(text, mode) for text in versions_text]

    aligned_versions_tokens = _align_recursively_token_lists(
        initial_token_lists, 
        min_versions_match, 
        min_lcs_len,
        PLACEHOLDER_TOKEN, # This is the padding placeholder
        recursion_depth=0,
        max_recursion_depth=max_recursion_depth
    )
    
    final_harmonized_tokens = _perform_voting_on_aligned_tokens(
        aligned_versions_tokens,
        PLACEHOLDER_TOKEN, # This is the padding placeholder
        tie_breaker_version_idx,
        tie_resolution_strategy,
        tie_placeholder_output # This is the output placeholder for ties
    )

    return _detokenize(final_harmonized_tokens, mode)

def test():
    # Open all versions of one document
    versions = []
    filename ="rsc37_rsc176_278.txt"
    current_dir = "/data/users/pfont"
    for folder in os.listdir(current_dir):
        folder_path = os.path.join(current_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("out_transcription_"):
            file_path = os.path.join(folder_path, filename)
            if os.path.exists(file_path):
                print(f"Found: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                versions.append(content)

    # Harmonize with default majority (3 out of 5)
    harmonized_text = harmonize_versions(versions)
    print(f"Harmonized (threshold 3/5):\n{harmonized_text}\n")


ocr_versions = ["hola.4 que tal estas era juan", "juan era aa hola que tal esca"]

harmonized_ocr_text = harmonize_documents(
    ocr_versions, 
    mode='char', 
    min_lcs_len=3, 
    min_versions_match=2, # Be explicit or let it default
    tie_resolution_strategy = "placeholder",
    tie_placeholder_output="_",
)
print(f"--- Harmonized OCR (char mode, min_lcs_len=7, min_match=2) ---\n'{harmonized_ocr_text}'\n")

# print(harmonize_versions(versions, 2))

