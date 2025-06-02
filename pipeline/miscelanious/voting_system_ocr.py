import re
from collections import Counter
import os, sys
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_final_output 

PLACEHOLDER_CHAR_IN_ALIGNMENT = '^' 

# --- Normalization Function ---
def normalize_text_for_document(text):
    """
    Normalizes a full document text: lowercase, remove form feeds, unify whitespace.
    """
    if not isinstance(text, str):
        text = str(text) # Ensure it's a string
    text = text.lower()
    text = text.replace('\f', '')  # Remove form feed characters
    # Consolidate multiple spaces that might result from \f removal or other sources, then strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Core Character-Level Alignment and Voting (for single lines) ---
# These functions are called by harmonize_document_per_line for each group of lines.

def _find_longest_common_substring_in_all(strings, min_lcs_len=1):
    """
    Finds the longest common substring present in ALL given strings.
    Helper for align_texts_recursively.
    Returns the LCS and a list of its starting indices, or (None, None).
    """
    if not strings or not all(s for s in strings):
        return None, None

    # Use the shortest non-empty string as the source for candidate substrings
    # Filter out empty strings before finding the shortest, then add them back for indexing if needed
    non_empty_strings_with_indices = [(s, idx) for idx, s in enumerate(strings) if s]
    if not non_empty_strings_with_indices: # All strings were empty
        return None, None

    # Sort by length to pick the shortest non-empty string
    non_empty_strings_with_indices.sort(key=lambda x: len(x[0]))
    shortest_str_data = non_empty_strings_with_indices[0]
    shortest_str_content = shortest_str_data[0]

    if len(shortest_str_content) < min_lcs_len:
        return None, None

    for length in range(len(shortest_str_content), min_lcs_len - 1, -1):
        for i in range(len(shortest_str_content) - length + 1):
            substring_candidate = shortest_str_content[i : i + length]
            
            indices = [-1] * len(strings)
            is_common_to_all_non_empty = True
            
            # Check if this candidate is in all *original* strings
            for k, s_to_check in enumerate(strings):
                if not s_to_check: # If original string is empty, it can't contain a non-empty substring
                    if substring_candidate: # If candidate is not empty, this string fails
                        is_common_to_all_non_empty = False
                        break
                    else: # Candidate is empty, empty string "contains" it at index 0
                        indices[k] = 0 
                else: # Original string is not empty
                    try:
                        indices[k] = s_to_check.index(substring_candidate)
                    except ValueError:
                        is_common_to_all_non_empty = False
                        break
            
            if is_common_to_all_non_empty:
                return substring_candidate, indices
                
    return None, None

def align_texts_recursively(texts, min_lcs_len=1):
    """
    Aligns a list of text strings (lines) recursively based on LCS.
    """
    if len(texts) <= 1:
        return texts

    if any(not t for t in texts if t is not None): # Check for empty strings, None is handled by caller
        active_texts = [t if t else "" for t in texts] # Treat None or empty as "" for alignment
    else:
        active_texts = list(texts)


    lcs, indices = _find_longest_common_substring_in_all(active_texts, min_lcs_len)

    if lcs is None: # No common substring found, or all strings became empty
        max_len = 0
        if any(active_texts): # Check if any string has content
             max_len = max(len(t) for t in active_texts)
        return [t.ljust(max_len, PLACEHOLDER_CHAR_IN_ALIGNMENT) for t in active_texts]

    aligned_parts = [""] * len(active_texts)
    
    left_segments = [active_texts[i][:indices[i]] for i in range(len(active_texts))]
    aligned_left_parts = align_texts_recursively(left_segments, min_lcs_len) if any(left_segments) else [""] * len(active_texts)

    right_segments = [active_texts[i][indices[i] + len(lcs):] for i in range(len(active_texts))]
    aligned_right_parts = align_texts_recursively(right_segments, min_lcs_len) if any(right_segments) else [""] * len(active_texts)
        
    for i in range(len(active_texts)):
        aligned_parts[i] = aligned_left_parts[i] + lcs + aligned_right_parts[i]
            
    return aligned_parts

def get_voted_string_from_aligned(aligned_texts):
    """
    Performs character-wise majority voting on a list of aligned text strings (lines).
    """
    if not aligned_texts or not aligned_texts[0]:
        return ""

    num_aligned_chars = len(aligned_texts[0])
    voted_chars = []

    for i in range(num_aligned_chars):
        column_chars = [text[i] for text in aligned_texts if i < len(text)] # Safety check
        if not column_chars: # Should not happen if alignment is correct
            voted_chars.append(PLACEHOLDER_CHAR_IN_ALIGNMENT)
            continue
        
        char_counts = Counter(column_chars)
        most_common_list = char_counts.most_common()
        
        if not most_common_list:
            voted_chars.append(PLACEHOLDER_CHAR_IN_ALIGNMENT)
            continue

        winner = most_common_list[0][0]
        max_count = most_common_list[0][1]

        # Tie-breaking: Prefer non-placeholder if tied
        if winner == PLACEHOLDER_CHAR_IN_ALIGNMENT and len(most_common_list) > 1:
            tied_non_placeholders = [
                char for char, count in most_common_list 
                if char != PLACEHOLDER_CHAR_IN_ALIGNMENT and count == max_count
            ]
            if tied_non_placeholders:
                # Tune: Tie-breaking for non-placeholders. Alphabetical is simple.
                # Could be based on order of OCR engines if confidence is known.
                winner = min(tied_non_placeholders) 
        voted_chars.append(winner)
            
    return "".join(voted_chars)

def harmonize_ocr_lines(ocr_lines_group, min_lcs_len=1):
    """
    Harmonizes a group of corresponding lines from multiple OCR outputs.
    This is the function that was previously named harmonize_ocr_outputs.
    """
    if not ocr_lines_group:
        return ""
    
    # Filter out None values that might have come from missing lines, treat as empty for alignment
    processed_lines_group = [line if line is not None else "" for line in ocr_lines_group]

    if all(not line for line in processed_lines_group): # All lines are empty or None
        return ""
    if len(processed_lines_group) == 1:
        return processed_lines_group[0]

    aligned_texts = align_texts_recursively(processed_lines_group, min_lcs_len)
    
    voted_text_raw = get_voted_string_from_aligned(aligned_texts)
    
    # Post-process the voted line:
    final_text = voted_text_raw.replace(PLACEHOLDER_CHAR_IN_ALIGNMENT, ' ')
    final_text = re.sub(r'\s+', ' ', final_text).strip()
    
    return final_text

# --- Noise Filtering (Heuristic) ---
def is_line_likely_garbage(line_text, min_alnum_char_count=3, min_alnum_ratio=0.4):
    """
    Heuristic to determine if a line is likely OCR noise.
    Tune: These thresholds (min_alnum_char_count, min_alnum_ratio) are critical and dataset-dependent.
          - min_alnum_char_count: Lines with fewer than this many alphanumeric chars might be noise,
                                  unless they are very short valid lines (e.g. "CO", "1.").
          - min_alnum_ratio: Ratio of alphanumeric characters to total characters. Low ratio means
                             many symbols/spaces relative to letters/numbers.
    Considerations:
        - Very short lines (e.g., "CO", "1.", page numbers) might be flagged incorrectly.
        - Lines that are purely punctuation (e.g., "---")
        - Lines with all caps vs mixed case (already handled by lowercasing in normalize)
    """
    if not line_text.strip(): # Completely empty or whitespace-only line is not garbage itself.
        return False

    alnum_count = sum(1 for char in line_text if char.isalnum())
    
    # Rule 1: Too few alphanumeric characters might indicate noise, especially if the line isn't extremely short.
    if alnum_count < min_alnum_char_count:
        # If the line is very short (e.g. <=3 chars), it might be a valid short string.
        # This check helps avoid flagging short valid items like "CO" or "1." as garbage too easily.
        if len(line_text.strip()) > min_alnum_char_count + 2: # e.g. if min_alnum_char_count is 3, check if len > 5
             return True # Longer line with very few alnum chars

    # Rule 2: Low ratio of alphanumeric characters
    # (Avoid division by zero for empty strings, though strip() above should handle it)
    if len(line_text) > 0 and (alnum_count / len(line_text)) < min_alnum_ratio:
        return True
        
    # Add more heuristics if needed:
    # - Too many consecutive identical characters (e.g., "aaaaaa")
    # - Presence of many unusual symbols not typical for the language
    
    return False

# --- Main Document Harmonization Function ---
def harmonize_full_document(list_of_full_ocr_texts,
                             char_level_min_lcs_len=1,
                             apply_noise_filter=True):
    """
    Harmonizes multiple OCR versions of a full document on a line-by-line basis.
    """
    if not list_of_full_ocr_texts:
        return ""
    
    # Ensure all inputs are strings
    list_of_full_ocr_texts = [str(text) if text is not None else "" for text in list_of_full_ocr_texts]

    if len(list_of_full_ocr_texts) == 1:
        return normalize_text_for_document(list_of_full_ocr_texts[0]) # Normalize even single doc

    # 1. Normalize each full document text
    normalized_ocr_docs = [normalize_text_for_document(doc_text) for doc_text in list_of_full_ocr_texts]

    # 2. Split each normalized document into lines
    doc_lines_list = [doc.splitlines() for doc in normalized_ocr_docs]

    # 3. Determine the maximum number of lines across all OCR versions
    max_doc_len_in_lines = 0
    if any(doc_lines_list): # Check if there's at least one list with lines
        # Filter out empty lists of lines before calling max, if any doc was completely empty
        non_empty_line_lists = [lines for lines in doc_lines_list if lines]
        if non_empty_line_lists:
            max_doc_len_in_lines = max(len(lines) for lines in non_empty_line_lists)
    
    if max_doc_len_in_lines == 0: # All input documents were empty or resulted in no lines after normalization
        return ""

    # 4. Iterate line by line, collect corresponding lines, and harmonize them
    final_harmonized_document_lines = []
    PLACEHOLDER_FOR_MISSING_LINE_INTERNAL = "" # Represent a missing line from an OCR as an empty string

    for line_idx in range(max_doc_len_in_lines):
        current_lines_for_this_index = []
        for ocr_doc_lines in doc_lines_list: # Iterate through each document's list of lines
            if line_idx < len(ocr_doc_lines):
                current_lines_for_this_index.append(ocr_doc_lines[line_idx])
            else:
                current_lines_for_this_index.append(PLACEHOLDER_FOR_MISSING_LINE_INTERNAL)
        
        # --- Optional Noise Filtering for the current group of lines ---
        lines_to_pass_to_harmonizer = []
        if apply_noise_filter:
            for line_text in current_lines_for_this_index:
                if line_text == PLACEHOLDER_FOR_MISSING_LINE_INTERNAL:
                    lines_to_pass_to_harmonizer.append(line_text) # Keep it as empty string
                elif is_line_likely_garbage(line_text):
                    # Tune: What to do with a garbage line?
                    # Option 1: Replace with empty string (reduces its voting power to placeholders)
                    lines_to_pass_to_harmonizer.append("")
                    # Option 2: Omit it (changes number of voters for this line, might be complex)
                    # Option 3: Pass it through (no filtering)
                else:
                    lines_to_pass_to_harmonizer.append(line_text)
        else:
            lines_to_pass_to_harmonizer = current_lines_for_this_index
        # --- End of Noise Filtering ---

        # Harmonize the collected (and possibly filtered) lines for the current line_idx
        # Ensure harmonize_ocr_lines can handle a list of all empty strings gracefully
        if not any(l for l in lines_to_pass_to_harmonizer): # If all are empty after filtering
            harmonized_line = ""
        else:
            harmonized_line = harmonize_ocr_lines(
                lines_to_pass_to_harmonizer,
                min_lcs_len=char_level_min_lcs_len
            )
        final_harmonized_document_lines.append(harmonized_line)

    # 5. Join the harmonized lines back into a single document string
    # Tune: Consider if filtering out completely empty lines from final_harmonized_document_lines
    #       is desirable before joining, or if original document structure (including empty lines)
    #       should be preserved as much as possible.
    #       Current approach keeps them, which might be closer to original formatting.
    return "\n".join(final_harmonized_document_lines)

def test(filename, output_folder="/home/pfont/pipeline/voting_system/voting_tests", current_dir = "/data/users/pfont"):
    # Open all versions of one document
    versions = []
    for folder in os.listdir(current_dir):
        folder_path = os.path.join(current_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("out_transcription_"):
            file_path = os.path.join(folder_path, filename)
            if "out_transcription_normal" not in folder:
                if os.path.exists(file_path):
                    print(f"Found: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    versions.append(content)
                    if folder == "out_transcription_binary_hisam_inverted":
                        versions.append(content)
                        versions.append(content)
    
    final_document_text = harmonize_full_document(
        versions,
        char_level_min_lcs_len=2,
        apply_noise_filter=True
    )
    save_final_output(final_document_text, filename, output_folder)


# --- Main Example ---
if __name__ == "__main__":
    # Load OCR versions from files
    ## Experiment with a couple of documents
    document_filenames = ["rsc37_rsc176_364.txt"]
    for filename in document_filenames:
        test(filename)