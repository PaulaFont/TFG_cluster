import re
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("es_core_news_lg")  

STOP_WORDS_CHUNKS = ["el", "la", "los", "las", "un", "una", "unos", "unas"]

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

def ner_function(text):
    """
    Extracts knowledge graph nodes by analyzing linguistic patterns.
    """
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
