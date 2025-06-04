import re
import os
import json, time
from collections import Counter
# from voting_utils import _detokenize # If you have it and it's better

INTERNAL_GAP_REPRESENTATION = object()
BASE_DIRECTORY = "/data/users/pfont/"


def tokenize_into_words_and_structure(text, preserve_case=True):
    if not isinstance(text, str): text = str(text)
    
    # Normalize newlines first to treat single and double newlines somewhat consistently
    # as potential paragraph separators for the regex.
    text = re.sub(r'\n\s*\n', '\n_PARAGRAPH_BREAK_\n', text) # Mark double newlines
    text = text.replace('\n', ' _LINE_BREAK_ ') # Mark single newlines

    if not preserve_case:
        text = text.lower()

    # Regex to capture:
    # - Words: [\w'-]+
    # - Markdown bold/italic markers: \*\*|\*|_
    # - Common section headers (case-insensitive for matching, token stores original)
    # - Bullets/numbers for lists: [-\*]|\d+[.)]
    # - Punctuation: [.,;:!?()]
    # - Our structural tokens: _PARAGRAPH_BREAK_ , _LINE_BREAK_
    # - Other symbols: [^\w\s]
    
    # Build a list of potential section headers (case-insensitive matching later)
    # Store them in a way that tokenizer can pick up their original casing
    raw_section_markers = [
        "RESULTANDO", "HECHOS PROBADOS", "CONSIDERANDO", "VISTOS",
        "FALLAMOS", "DECRETAMOS", "DESEAMOS", "SENTENCIA", "PRESIDENTE",
        "VOCALES", "ASESOR", "DEFENSOR DEL PROCESADO", "SOLICITUD",
        "SULTANDO", "ACUERDO", "ORDEN", "TESTIMONIO" # Add more as observed
    ]
    # Create a regex part for section markers, ensuring to capture variations
    # This is simplified; a more robust way would be to find these first, replace with a unique token,
    # then tokenize the rest.
    section_marker_regex_parts = []
    for marker in raw_section_markers:
        # Create a regex that matches the marker, possibly followed by a colon, and is case-insensitive for the match
        # but we want to capture the original cased token.
        # This is tricky with re.findall. A simpler approach might be to normalize markers after tokenization.
        section_marker_regex_parts.append(re.escape(marker) + r":?")


    # Simpler tokenization: split by space, then refine. Less prone to complex regex issues.
    # First, protect newlines by replacing them with unique placeholders
    text = text.replace("\n_PARAGRAPH_BREAK_\n", " __PARABREAK__ ")
    text = text.replace("_LINE_BREAK_", " __LINEBREAK__ ")

    # Tokenize by word characters, numbers, and then treat everything else as individual tokens.
    # This is generally more robust than one massive regex.
    pre_tokens = re.findall(r"[\w'-]+|[.,;:!?()]|\*\*|--+|==|[-\*]|\d+[.)]|[^\s\w]", text)
    
    tokens = []
    for pt in pre_tokens:
        if pt == "__PARABREAK__":
            tokens.append("_PARAGRAPH_BREAK_")
        elif pt == "__LINEBREAK__":
            tokens.append("_LINE_BREAK_")
        else:
            tokens.append(pt)
            
    return [t for t in tokens if t.strip()]


def detokenize_words_with_structure(tokens):
    if not tokens: return ""
    
    output_parts = []
    for i, token in enumerate(tokens):
        if token == "_PARAGRAPH_BREAK_":
            output_parts.append("\n\n")
            continue
        if token == "_LINE_BREAK_":
            if output_parts and output_parts[-1] != "\n\n" and output_parts[-1] != "\n":
                 output_parts.append("\n")
            continue

        if not output_parts or output_parts[-1] in ["\n\n", "\n"]:
            output_parts.append(token)
        else:
            prev_token = tokens[i-1] # This needs to be the previously *appended text token*
            # Find last actual text token, not structural token
            last_text_token = ""
            for j in range(len(output_parts) - 1, -1, -1):
                if output_parts[j] not in ["\n\n", "\n"]:
                    last_text_token = output_parts[j]
                    break
            
            # Basic spacing (can be significantly improved)
            if token in ".,;:!?)'" or \
               (token.startswith("'") and token.lower() in ["'s", "'re", "'ll", "'ve", "'m", "'d"]):
                # Remove preceding space if it exists before appending punctuation
                if output_parts and output_parts[-1] == " ":
                    output_parts.pop()
                output_parts.append(token)
            elif last_text_token and last_text_token in "([\"“": # No space after opening brackets/quotes
                 output_parts.append(token)
            elif token in ")]\"”" and last_text_token and last_text_token not in "([\"“":
                if output_parts and output_parts[-1] == " ": output_parts.pop()
                output_parts.append(token)
            else:
                # Add space only if the last part isn't already a space-like newline
                if not (output_parts and output_parts[-1].endswith("\n")):
                     output_parts.append(" ")
                output_parts.append(token)
    
    text = "".join(output_parts)
    # Cleanup common markdown or extra spaces around punctuation
    text = re.sub(r'\s+([.,;:!?)"])', r'\1', text) 
    text = re.sub(r'([(["“])\s+', r'\1', text)    
    text = text.replace(" n't", "n't") # etc.
    text = re.sub(r'\n\s*\n(\s*\n)+', '\n\n', text) # Consolidate multiple blank lines
    return text.strip()


def harmonize_llm_outputs_word_level_final(llm_outputs,
                                           preserve_case_in_voting=True,
                                           min_votes_to_include=2,
                                           layout_aware_indices=None): # List of indices for layout-aware versions
    """
    Harmonizes multiple LLM outputs using direct word-column voting.
    - `preserve_case_in_voting`: If True, 'Word' and 'word' are different for voting.
    - `min_votes_to_include`: A token must receive at least this many votes in a column
                             to be included in the final output.
    - `layout_aware_indices`: Indices of LLM outputs that came from layout-analyzed OCR.
                              Used for tie-breaking.
    """
    if not llm_outputs: return ""
    
    num_llm_versions = len(llm_outputs)
    if num_llm_versions == 0: return ""

    processed_outputs = [str(out) if out is not None else "" for out in llm_outputs]
    if num_llm_versions == 1: return processed_outputs[0]

    # Tokenize using the structure-aware tokenizer
    tokenized_outputs = [tokenize_into_words_and_structure(doc, preserve_case=preserve_case_in_voting) for doc in processed_outputs]

    max_word_count = 0
    if any(tokenized_outputs):
        non_empty_token_lists = [tl for tl in tokenized_outputs if tl]
        if non_empty_token_lists:
            max_word_count = max(len(tokens) for tokens in non_empty_token_lists)
    if max_word_count == 0: return ""

    # Adjust min_votes_to_include based on your rule "wins at 2 if text is equal"
    # This needs careful consideration. If 2 versions agree perfectly on a long stretch,
    # this column-by-column voting won't directly capture that.
    # The `min_votes_to_include` here is per-column.
    # For "wins at 2", we'll use it as an absolute minimum.
    if not isinstance(min_votes_to_include, int) or min_votes_to_include < 1:
        min_votes_to_include = 1 # Default
    # Ensure threshold isn't higher than number of versions
    min_votes_to_include = min(min_votes_to_include, num_llm_versions)


    final_voted_words = []
    for word_idx in range(max_word_count):
        current_word_column_with_source = [] # Store (token, source_index)
        active_inputs_for_column = 0
        for i, token_list in enumerate(tokenized_outputs):
            if word_idx < len(token_list):
                current_word_column_with_source.append((token_list[word_idx], i))
                active_inputs_for_column +=1
            else:
                current_word_column_with_source.append((INTERNAL_GAP_REPRESENTATION, i))
        
        if active_inputs_for_column == 0: continue

        # Get tokens for counting, excluding internal gaps
        column_tokens_for_counting = [
            item[0] for item in current_word_column_with_source 
            if item[0] != INTERNAL_GAP_REPRESENTATION
        ]

        if not column_tokens_for_counting: # Only internal gaps in this column
            continue

        word_counts = Counter(column_tokens_for_counting)
        most_common_candidates = word_counts.most_common()
        
        if not most_common_candidates: continue

        top_candidate_word, top_candidate_count = most_common_candidates[0]

        if top_candidate_count >= min_votes_to_include:
            tied_winners_with_source_info = []
            for token_val, source_idx in current_word_column_with_source:
                if token_val != INTERNAL_GAP_REPRESENTATION and word_counts[token_val] == top_candidate_count:
                    is_layout_aware = layout_aware_indices and source_idx in layout_aware_indices
                    # Store (token, is_layout_aware_source) to help with tie-breaking
                    # Avoid adding duplicates of the same token if it appears multiple times from same-quality sources
                    # Check if (token_val, is_layout_aware) is already in list before appending
                    if not any(tw[0] == token_val and tw[1] == is_layout_aware for tw in tied_winners_with_source_info):
                         tied_winners_with_source_info.append( (token_val, is_layout_aware) )


            if not tied_winners_with_source_info : continue # Should not happen if candidates existed

            if len(tied_winners_with_source_info) == 1:
                chosen_word = tied_winners_with_source_info[0][0]
            else:
                # Tie-breaking:
                # 1. Prefer tokens from layout-aware sources if `layout_aware_indices` is provided.
                # 2. Then, alphabetical.
                if layout_aware_indices:
                    layout_aware_tied = [tw[0] for tw in tied_winners_with_source_info if tw[1]]
                    if layout_aware_tied:
                        chosen_word = sorted(layout_aware_tied)[0]
                    else: # No layout-aware sources among ties, pick alphabetically from all tied
                        chosen_word = sorted([tw[0] for tw in tied_winners_with_source_info])[0]
                else: # No layout preference, pick alphabetically
                    chosen_word = sorted([tw[0] for tw in tied_winners_with_source_info])[0]
            final_voted_words.append(chosen_word)
            
    harmonized_text = detokenize_words_with_structure(final_voted_words)
    return harmonized_text

# --- Main Example ---
if __name__ == "__main__":
    base_directory = BASE_DIRECTORY
    json_path = os.path.join(base_directory , "llm_versions.json") 

    if not os.path.exists(json_path):
        print(f"Error: llm_versions.json not found at {json_path}")
        # Fallback dummy data for testing
        data_dict = {
            "566": {
                "llm_version_info": ["v_normal_1", "v_normal_2", "v_cut_1", "v_cut_2"],
                "llm_version_text": [
                    "**HECHOS**\n\nPrimer hecho importante.",
                    "**HECHOS PROBADOS**\n\nPrimer hecho.\nSegundo hecho.",
                    "Hechos: Primer hecho importante.",
                    "Hechos Probados\n\nPrimer hecho.\nUn segundo hecho."
                ]
            }
        }
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)

    doc_key_to_process = "364" # Change this to test other documents in your JSON
    llm_outputs_all = data_dict.get(doc_key_to_process, {}).get("llm_version_text", [])
    llm_info_all = data_dict.get(doc_key_to_process, {}).get("llm_version_info", [])

    if not llm_outputs_all:
        print(f"No LLM versions found for key '{doc_key_to_process}'.")
    else:
        print(f"Processing {len(llm_outputs_all)} LLM versions for document '{doc_key_to_process}'.")

        # Identify indices of layout-aware versions ("cut" versions)
        layout_aware_indices = [i for i, info in enumerate(llm_info_all) if "cut" in info.lower()]
        print(f"Layout-aware version indices: {layout_aware_indices}")

        print("\n--- Harmonizing LLM Outputs (Final Refined Direct Column Voting) ---")
        start_time = time.time()
        
        # Your "wins at 2" requirement:
        # This threshold is crucial. For 10 versions, a threshold of 2 means a word
        # is included if just 20% of the LLMs agree on it, even if the other 80% have
        # a different consensus or gaps. This might be too low for diverse outputs.
        # Consider a majority: (len(llm_outputs_all) // 2) + 1
        # Or a percentage: int(len(llm_outputs_all) * 0.4) # 40% agreement
        threshold = 2 # As per your specific "wins at 2"

        harmonized_result = harmonize_llm_outputs_word_level_final(
            llm_outputs_all,
            preserve_case_in_voting=True, 
            min_votes_to_include=threshold,
            layout_aware_indices=layout_aware_indices
        )
        
        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.4f} seconds")

        print("\n--- Final Harmonized Text ---")
        print(harmonized_result)

        # Example of saving:
        output_filename = f"harmonized_{doc_key_to_process}_final.txt"
        with open(output_filename, "w", encoding="utf-8") as outfile:
            outfile.write(harmonized_result)
        print(f"\nHarmonized text saved to {output_filename}")