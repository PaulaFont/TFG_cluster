import re
import os
import json, time
from collections import Counter
from difflib import SequenceMatcher # For pairwise alignment

# --- Configuration ---
# Internal representation for gaps created by the alignment algorithm.
# This should NOT appear in the final output.
MSA_GAP_TOKEN = "<MSA_GAP>"
BASE_DIRECTORY = "/data/users/pfont/"
JSON_PATH = os.path.join(BASE_DIRECTORY , "llm_versions.json") 
OUTPUT_FOLDER = os.path.join(BASE_DIRECTORY , "out_final_transcription") 

# --- Tokenization and Detokenization ---
def tokenize_into_words(text, preserve_case=True):
    if not isinstance(text, str): text = str(text)
    if not preserve_case:
        text = text.lower()
    # Keeps words (alphanumeric + apostrophes/hyphens within words) and most punctuation separate.
    tokens = re.findall(r"[\w'-]+|\*\*---|==|[.,;:!?()]|[^\w\s]", text)

    refined_tokens = []
    for token in tokens:
        if not token: continue
        if len(token) > 1 and token[-1] in '.,;:!?' and token[:-1].isalnum():
            refined_tokens.append(token[:-1])
            refined_tokens.append(token[-1])
        else:
            refined_tokens.append(token)
            
    return [t for t in refined_tokens if t.strip()]

def detokenize_words(tokens):
    if not tokens: return ""
    text = ""
    for i, token in enumerate(tokens):
        if i == 0:
            text = token
        else:
            prev_token = tokens[i-1]
            # Avoid space before common punctuation
            if token in ".,;:!?)'" or \
               (token.startswith("'") and token.lower() in ["'s", "'re", "'ll", "'ve", "'m", "'d"]):
                text += token
            # Avoid space after opening brackets/quotes
            elif prev_token in "([\"“":
                 text += token
            # Avoid space before closing if previous was not an opening bracket/quote
            elif token in ")]\"”" and prev_token not in "([\"“":
                 if len(text) > 0 and text[-1] == ' ': text = text[:-1]
                 text += token
            else: # Default: add a space
                text += " " + token
    
    # General cleanup for multiple spaces that might have been introduced
    text = re.sub(r'\s+', ' ', text).strip()
    # Specific cleanups
    text = text.replace(" @-@ ", "-") # For hyphenated words split by tokenizer
    text = text.replace(" --- \n", "---\n") # Markdown horizontal rule
    return text

# --- Pairwise Word Sequence Alignment using difflib ---
def align_two_word_lists(list1, list2, gap_token=MSA_GAP_TOKEN):
    """
    Aligns two lists of word tokens using difflib.SequenceMatcher.
    Returns two new lists of the same length, padded with gap_token.
    """
    sm = SequenceMatcher(None, list1, list2, autojunk=False)
    aligned1, aligned2 = [], []
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            aligned1.extend(list1[i1:i2])
            aligned2.extend(list2[j1:j2])
        elif tag == 'replace':
            len1 = i2 - i1
            len2 = j2 - j1
            aligned1.extend(list1[i1:i2])
            aligned2.extend(list2[j1:j2])
            # Pad the shorter segment of the replacement
            if len1 < len2:
                aligned1.extend([gap_token] * (len2 - len1))
            elif len2 < len1:
                aligned2.extend([gap_token] * (len1 - len2))
        elif tag == 'delete': # list1 has words, list2 has gaps
            aligned1.extend(list1[i1:i2])
            aligned2.extend([gap_token] * (i2 - i1))
        elif tag == 'insert': # list2 has words, list1 has gaps
            aligned1.extend([gap_token] * (j2 - j1))
            aligned2.extend(list2[j1:j2])
    return aligned1, aligned2

def harmonize_llm_outputs_iterative_consensus(llm_outputs,
                                            preserve_case=True,
                                            debug=False):
    if not llm_outputs: return ""

    num_llm_versions = len(llm_outputs)
    if num_llm_versions == 0: return ""
    processed_outputs = [str(out) if out is not None else "" for out in llm_outputs]
    if num_llm_versions == 1: return processed_outputs[0]

    tokenized_docs = [tokenize_into_words(doc, preserve_case=preserve_case) for doc in processed_outputs]

    consensus_tokens = []
    reference_idx = -1
    for i, tokens in enumerate(tokenized_docs):
        if tokens:
            consensus_tokens = list(tokens) # Make a copy
            reference_idx = i
            break

    if reference_idx == -1: # All documents were empty after tokenization
        return ""

    if debug: print(f"Initial consensus (from doc {reference_idx+1}): {' '.join(consensus_tokens)}")

    for i in range(num_llm_versions):
        if i == reference_idx or not tokenized_docs[i]: # Skip the reference itself or empty docs
            continue

        if debug: print(f"\nAligning doc {i+1} ('{llm_outputs_to_process[i][:30]}...') to current consensus...")

        current_doc_tokens = tokenized_docs[i]
        aligned_consensus, aligned_doc = align_two_word_lists(consensus_tokens, current_doc_tokens)

        if debug:
            # Print full aligned sequences for better debugging
            print(f"  Consensus aligned (len {len(aligned_consensus)}): {' '.join(aligned_consensus)}")
            print(f"  Doc {i+1} aligned (len {len(aligned_doc)}):    {' '.join(aligned_doc)}")

        new_consensus_tokens = []
        for c_idx, (c_tok, d_tok) in enumerate(zip(aligned_consensus, aligned_doc)):
            column = [c_tok, d_tok]
            actual_toks_in_col = [t for t in column if t != MSA_GAP_TOKEN]

            if not actual_toks_in_col: # Both were MSA_GAP_TOKEN
                # This column remains a gap in the consensus, so we don't add any token
                pass
            elif len(actual_toks_in_col) == 1:
                # One is a gap, the other is a token. Choose the token.
                new_consensus_tokens.append(actual_toks_in_col[0])
            else: # Both are actual tokens (c_tok and d_tok are not MSA_GAP_TOKEN)
                if c_tok == d_tok: # Case-sensitive comparison (due to preserve_case=True earlier)
                    new_consensus_tokens.append(c_tok)
                else:
                    # --- MODIFICATION START ---
                    # Conflict: Both actual tokens, but different.
                    # Option 1: Prefer current consensus token (More stable)
                    new_consensus_tokens.append(c_tok)
                    if debug and c_idx < 20: # Print first few conflicts
                         print(f"    Conflict at index {c_idx}: Consensus='{c_tok}', Doc='{d_tok}'. Chose Consensus.")
                    # Option 2: Prefer new document's token (More adaptive to new info)
                    # new_consensus_tokens.append(d_tok)
                    # if debug and c_idx < 20:
                    #      print(f"    Conflict at index {c_idx}: Consensus='{c_tok}', Doc='{d_tok}'. Chose Doc.")
                    # Option 3: Alphabetical (arbitrary but deterministic)
                    # chosen_tok = sorted([c_tok, d_tok])[0]
                    # new_consensus_tokens.append(chosen_tok)
                    # if debug and c_idx < 20:
                    #      print(f"    Conflict at index {c_idx}: Consensus='{c_tok}', Doc='{d_tok}'. Chose '{chosen_tok}' (alphabetical).")
                    # Option 4 (Original problematic one): Omit on conflict by 'pass'
                    # pass
                    # --- MODIFICATION END ---


        consensus_tokens = new_consensus_tokens
        if debug:
            print(f"  New consensus (len {len(consensus_tokens)}):     {' '.join(consensus_tokens)}")
        if not consensus_tokens and debug: print("  Consensus became empty!")


    if debug: print(f"\nFinal consensus token list (len {len(consensus_tokens)}): {' '.join(consensus_tokens)}")
    final_text = detokenize_words(consensus_tokens)
    return final_text

def one_document(data_dict, d_id, out_folder, mode):

    llm_outputs_to_process = data_dict.get(d_id, {}).get("llm_version_text", [])
    filename = f"rsc37_rsc176_{d_id}_{mode}.txt"
    out_path = os.path.join(out_folder , filename) 

    match mode:
        case "no_layout":
            llm_outputs_to_process = llm_outputs_to_process[0:6]
        case "layout":
            llm_outputs_to_process = llm_outputs_to_process[6:]
    
    print(f"Processing {len(llm_outputs_to_process)} LLM versions for document '{d_id}'.")

    harmonized_result = harmonize_llm_outputs_iterative_consensus(
            llm_outputs_to_process,
            preserve_case=True,
            debug=False
        )

    with open(out_path, "w", encoding="utf-8") as outfile:
        outfile.write(harmonized_result)

    print(f"\nHarmonized text saved to {out_path}")

    
# --- Main Example ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
    mode = "no_layout"
    for key in data_dict.keys():
        one_document(data_dict, key, OUTPUT_FOLDER, mode)
