# ner_logic.py
# Utilities for extracting named entities and relevant phrases from Spanish text using spaCy and custom patterns.
import re
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("es_core_news_lg")  

STOP_WORDS_CHUNKS = ["el", "la", "los", "las", "un", "una", "unos", "unas"]

# For global
"Returns True if we keep it the same, returns False if it's specific"
def filter_nodes_global(text):
    def is_date(text):
        year_pattern = r"^[12]\d{3}$"
        date_month_year_pattern = r"^\d{1,2}\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}$"
        if re.fullmatch(year_pattern, text):
            return True
        if re.fullmatch(date_month_year_pattern, text, re.IGNORECASE):
            return True
        return False

    if is_date(text):
        return True, "DATE"

    global nlp
    doc = nlp(text)
    for ent in doc.ents:
        label = ent.label_
        if label in ["PER", "PERSON"]:
            return True, "PERSON"
        elif label in ["LOC", "GPE"]:
            return True, "LOCATION"
        elif label == "DATE":
            return True, "DATE"
    return False, "OTHER"

# In general
def extract_spacy_ents(doc):
    return {ent.text.strip() for ent in doc.ents}

def add_pattern_matches(doc, matcher):
    entities = set()
    matches = matcher(doc)
    for _, start, end in matches:
        entities.add(doc[start:end].text.strip())
    return entities

def build_matcher(nlp):
    matcher = Matcher(nlp.vocab)
    matcher.add("DATE_FULL", [[{"IS_DIGIT": True}, {"LOWER": "de"}, {"IS_ALPHA": True}, {"LOWER": "de"}, {"IS_DIGIT": True}]])
    matcher.add("DATE_PARTIAL", [[{"IS_DIGIT": True}, {"LOWER": "de"}, {"IS_ALPHA": True}]])
    matcher.add("DATE_MONTH_YEAR", [[{"IS_ALPHA": True}, {"LOWER": "de"}, {"IS_DIGIT": True}]])
    matcher.add("AGE_DURATION", [[{"LIKE_NUM": True}, {"LOWER": "años"}]])
    matcher.add("LEGAL_ARTICLE", [[{"LOWER": "artículo"}, {"IS_DIGIT": True}]])
    return matcher

def extract_noun_chunks(doc):
    entities = set()
    for chunk in doc.noun_chunks:
        if len(chunk) > 1 and chunk[-1].pos_ == "ADJ" and not chunk[-1].is_title:
            final_chunk = chunk[:-1]
        else:
            final_chunk = chunk
        chunk_text = final_chunk.text.strip()
        first_token_lower = chunk_text.split()[0].lower() if chunk_text else ''
        if first_token_lower in STOP_WORDS_CHUNKS:
            chunk_text = ' '.join(chunk_text.split()[1:])
        if len(chunk_text) > 2 and chunk_text.lower() not in STOP_WORDS_CHUNKS:
            entities.add(chunk_text)
    return entities

def extract_tokens(doc):
    entities = set()
    for token in doc:
        if token.like_num and len(token.text) == 4 and token.text.startswith(('18', '19', '20')):
            entities.add(token.text.strip())
        if token.is_upper and len(token.text) > 1:
            entities.add(token.text.strip())
        if token.pos_ == "PROPN" and token.is_title:
            entities.add(token.text.strip())
        if token.pos_ == "NOUN" and token.dep_ in ("nsubj", "obj", "pobj", "dobj"):
            entities.add(token.lemma_.lower())
    return entities

def remove_redundant_entities(entities):
    final_entities = set()
    sorted_entities = sorted(list(entities), key=len, reverse=True)
    for ent in sorted_entities:
        is_substring = False
        for final_ent in final_entities:
            if f" {ent} " in f" {final_ent} " or final_ent.startswith(f"{ent} ") or final_ent.endswith(f" {ent}"):
                if re.fullmatch(r"1[89]\d{2}|20\d{2}", ent):
                    continue
                is_substring = True
                break
        if not is_substring:
            final_entities.add(ent)
    return {e for e in final_entities if e.lower() not in STOP_WORDS_CHUNKS and len(e) > 1}

def postprocess_entities(entities):
    processed = set()
    for ent in entities:
        ent = re.sub(r'\s*\([^)]*\)$', '', ent).strip()
        if ' y ' in ent and ent.count(' ') > 3:
            parts = ent.split(' y ')
            if all(p[0].isupper() for p in parts):
                processed.update(parts)
                continue
        processed.add(ent)
    return processed

#TODO: Maybe add a separation to dates in simple?
def ner_function(text, simple=True):
    """
    Extracts knowledge graph nodes by analyzing linguistic patterns. Two modes: simple and complex.
    """
    if simple:
        doc = nlp(text)
        named_entities = extract_spacy_ents(doc)
        print(f"For text {text} the NER found {named_entities}")
        return list(named_entities)

    # Complex function. Separates with more detail (year, words, ... )
    doc = nlp(text)
    matcher = build_matcher(nlp)
    entities = set()
    entities |= extract_spacy_ents(doc)
    entities |= add_pattern_matches(doc, matcher)
    entities |= extract_noun_chunks(doc)
    entities |= extract_tokens(doc)
    entities = remove_redundant_entities(entities)
    entities = postprocess_entities(entities)
    return sorted(list(entities))


def link_components_by_context(original_phrase: str, substrings: list[str]) -> list[tuple[str, str, str]]:
    """
    Creates a relational chain of triplets by analyzing the context within an original phrase.

    This function reconstructs the relationship between component parts by:
    1. Finding the start position of each substring in the original phrase.
    2. Sorting the substrings by their appearance order.
    3. For each adjacent pair, extracting the intermediate text (the "glue") to use as the predicate.
    
    This replaces a single complex node with a more expressive, interconnected graph of its parts.

    Args:
        original_phrase: The full, original string from which the substrings were derived.
        substrings: A list of the component nodes.

    Returns:
        A list of triplets in the format (head_node, tail_node, relationship).
    """
    if not substrings:
        return []

    # Step 1: Find the position of each substring in the original phrase.
    # We store them to sort them and avoid re-calculating indices.
    found_nodes = []
    # Use a copy to safely remove items
    searchable_substrings = sorted(substrings, key=len, reverse=True)
    temp_phrase = original_phrase.lower()

    for sub in searchable_substrings:
        try:
            start_index = temp_phrase.find(sub.lower())
            if start_index != -1:
                found_nodes.append({'text': sub, 'start': start_index, 'end': start_index + len(sub)})
                # Blank out the found part to avoid re-matching a substring (e.g., 'pena' in 'pena menor')
                temp_phrase = temp_phrase[:start_index] + (' ' * len(sub)) + temp_phrase[start_index + len(sub):]
        except ValueError:
            continue

    # Step 2: Sort the found nodes by their start position in the original phrase.
    ordered_nodes = sorted(found_nodes, key=lambda x: x['start'])
    
    # Step 3: Create a chain of triplets based on the determined order.
    triplets = []
    if len(ordered_nodes) < 2:
        return []

    for i in range(len(ordered_nodes) - 1):
        head_node_info = ordered_nodes[i]
        tail_node_info = ordered_nodes[i+1]
        
        # The relationship is the text between the end of the first node and the start of the next.
        start_of_relation = head_node_info['end']
        end_of_relation = tail_node_info['start']
        
        relation_text = original_phrase[start_of_relation:end_of_relation].strip()
        
        # Clean up the relation to be a valid, simple predicate
        if not relation_text:
            relation_text = 'is_related_to'

        triplets.append((head_node_info['text'], tail_node_info['text'], relation_text))
        
    return triplets


# print(link_components_by_context( '', ['Rebelión', 'auxilio', 'delito']))

def separate_node_y(ent):
    ent = re.sub(r'\s*\([^)]*\)$', '', ent).strip()
    if ' y ' in ent:
        parts = ent.split(' y ')
        if all(p[0].isupper() for p in parts):
            # Batch process all parts at once
            doc = nlp('. '.join(parts))
            person_set = {ent.text for ent in doc.ents if ent.label_ in ["PER", "PERSON", "LOC", "GPE"]}
            if all(p in person_set for p in parts):
                return parts
    return [ent]

def preprocess_triplets(triplets):
    """
    Expande nodos compuestos (como 'Juan y Sara') en todas las tripletas,
    propagando los cambios a todas las tripletas que los referencian.
    """
    # 1. Mapping all nodes to the new ones
    node_map = {}
    for s, p, o in triplets:
        for node in [s, o]:
            if node not in node_map:
                split = separate_node_y(node)
                node_map[node] = split

    # 2. Expand nodes using mapping
    new_triplets = []
    for s, p, o in triplets:
        subjects = node_map.get(s, [s])
        objects = node_map.get(o, [o])
        for subj in subjects:
            for obj in objects:
                new_triplets.append((subj, p, obj))
    return new_triplets

