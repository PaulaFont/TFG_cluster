import os
import re
import networkx as nx
import pickle
import spacy
import Levenshtein as levenshtein
from num2words import num2words
import math
from ner_logic import ner_function

MAX_LEN_NODE = 6 #(words)
MAX_LEN_EDGE = 5 #(words)
GRAPH_DIRECTORY = "/data/users/pfont/graph/"
KG_FILENAME = "online_knowledge_graph_tests.pkl" 



# Apply NER in here. (Future idea: to provide information to the node as, the type: PER, LOC, ORG, DATE, ...)
def manage_wrongs(list_text):
    new_triplets = []
    for text in list_text:
        entities = [ent.text for ent in nlp(text).ents if ent is not None]
        entity_positions = []
        for e in entities:
            match = re.search(re.escape(e), text)
            if match:
                entity_positions.append((e, match.start()))
            else:
                print(f"Warning: entity '{e}' not found in text.")

        entity_positions.sort(key=lambda x: x[1])

        triplets = []
        current_subject = None

        for i in range(len(entity_positions) - 1):
            s_entity, s_idx = entity_positions[i]
            o_entity, o_idx = entity_positions[i + 1]

            # If no subject yet, assume first entity is the subject
            if current_subject is None:
                current_subject = s_entity

            # Extract relation
            relation = text[s_idx + len(s_entity):o_idx].strip(" ,.")

            # Check if the relation contains a verb or makes grammatical sense
            if any(verb in relation for verb in ['was', 'served', 'became', 'is', 'are']):
                triplets.append((current_subject, relation, o_entity))

                # Update subject if it's a new clause with a new actor
                if 'and' in relation or ',' in relation:
                    current_subject = current_subject  # optional: update if needed
        new_triplets.append(triplets)


# DONE
def are_strings_similar(word1, word2):
    """
    Determines whether two strings are similar based on a normalized Levenshtein distance.

    This function compares two input strings by:
      1. Converting any numeric digits to their textual representation in Spanish.
      2. Normalizing the strings by converting to lowercase and removing all non-letter characters.
      3. Calculating the normalized Levenshtein distance between the processed strings.
      4. Returning True if the distance is within an allowed threshold, which is dynamically determined based on the string length.

    Args:
        word1 (str): The first string to compare.
        word2 (str): The second string to compare.

    Returns:
        bool: True if the strings are considered similar, False otherwise.
    """
    # 0. Calcular cuantos errores se permiten
    def max_allowed_errors(n: int) -> int:
        # Número máximo de errores permitidos para una cadena de longitud n.
        return max(1, int(math.log2(n)))

    # 1. Convert all numbers in string to text ("6" becomes "six")
    def numeros_a_texto(texto: str) -> str:
        # Reemplaza todos los números encontrados por su versión en texto (en español)
        return re.sub(r'\d+', lambda m: num2words(int(m.group()), lang='es'), texto)

    s1 = numeros_a_texto(word1)
    s2 = numeros_a_texto(word2)

    # 2. Normalize text. Remove everything that's not letters. (Dots, noise, ...)
    def normalise(s):
        """lower-case and strip out every character that is not a-z."""
        return re.sub(r'[^a-z]', '', s.lower())

    s1 = normalise(numeros_a_texto(word1))
    s2 = normalise(numeros_a_texto(word2))
    if not s1 or not s2:
        return False # We don't compare symbols, empty strings, ... 

    # 3. Get Numeric Distance between two processed strings
    max_len = max(len(s1), len(s2))
    # threshold = 2 / max_len if max_len > 4 else 1 / max_len
    threshold = max_allowed_errors(max_len) / max_len
    dist = levenshtein.distance(s1, s2) / max_len
    return dist <= threshold

# Empty function. Apply are_strings_similar in here
def compare_to_existing(current_graph, triplets_to_add):
    existing_nodes = current_graph.nodes()


# Idea: to manage long edges somehow (AT THE MOMENT WE WILL SKIP THEM)    
def create_intermediate_triplets(subject: str, long_predicate: str, obj: str):# -> List[Tuple[str, str, str]]:
    """Create intermediate nodes/triplets for long predicates"""
    
    # Try to extract entities from the long predicate
    entities = [ent.text for ent in nlp(long_predicate).ents if ent is not None]

    
    if entities:
        # Use entities as intermediate nodes
        triplets = []
        current_subj = subject
        
        for i, entity in enumerate(entities):
            if i == len(entities) - 1:
                # Last entity connects to final object
                relation = "leads_to"
                triplets.append((current_subj, relation, obj))
            else:
                # Connect through intermediate entity
                relation = "connected_via"
                triplets.append((current_subj, relation, entity))
                current_subj = entity
        
        return triplets
    
    # Try to split by conjunctions and create chain
    parts = re.split(r'\s+and\s+|\s*,\s+|\s+then\s+', long_predicate)
    parts = [p.strip() for p in parts if p.strip()]
    
    if len(parts) > 1:
        triplets = []
        current_subj = subject
        
        for i, part in enumerate(parts):
            short_part = shorten_with_ner(part, MAX_LEN_EDGE)
            
            if i == len(parts) - 1:
                # Last part connects to final object
                triplets.append((current_subj, short_part, obj))
            else:
                # Create intermediate node
                intermediate_node = f"intermediate_{hash(part) % 1000}"
                triplets.append((current_subj, short_part, intermediate_node))
                current_subj = intermediate_node
        
        return triplets
    
    # Fallback: just shorten the predicate
    short_predicate = shorten_with_ner(long_predicate, MAX_LEN_EDGE)
    return [(subject, short_predicate, obj)]

# DONE    
def filter_edge(edge):
    """
    Checks if the number of words in the edge string is less than MAX_LEN_EDGE.
    """
    return len(edge.split()) < MAX_LEN_EDGE

# General function.  #TODO: needs redoing
def filter_and_fix_triplets(current_graph, triplets):
    new_nodes = []

    adding_nodes = [x for s, p, o in triplets for x in (s, o)]
    adding_edges = [p for s, p, o in triplets]

    # Filter by length (nodes and edges)
    filtered_nodes = [node for node in adding_nodes if len(node) < MAX_LEN_NODE] # Filter for short nodes
    filtered_edges = [edge for edge in adding_edges if len(edge) < MAX_LEN_EDGE] # Filter for short edges

    # Do NER to long triplets and divide them
    long_nodes = list(set(adding_nodes).symmetric_difference(set(filtered_nodes))) # All new nodes that are not in short nodes list
    triplets_with_long_nodes = [(s,p,o) for (s,p,o) in triplets if s or o in long_nodes]
    long_edges = list(set(adding_edges) - set(filtered_edges))
    triplets_with_long_edges = [(s,p,o) for (s,p,o)  in triplets if p in long_edges]

    

    # String distance to the nodes in the graph (only letters, put in lowercase, normalize by text_length ??)
    # TODO: do



# ------------------------------------------------------------------------
# End of filtering/fixing helpers, start of LLM extraction and graph update
# ------------------------------------------------------------------------



def extract_triplets(text_content, client, model="microsoft/phi-4"):
    if not text_content or not client:
        return False

    messages_for_llm, parser_func = produce_prompt_for_kg_extraction(text_content)
    
    print(f"\nAttempting to extract KG triples from text (length: {len(text_content)} chars)...")

    try:
        # query_llm uses the OpenAI client directly with messages
        completion = client.chat.completions.create(
            model=model,
            messages=messages_for_llm,
            temperature=0.2
        )
        raw_llm_response = completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling LLM for KG triple extraction: {e}")
        return False

    if not raw_llm_response:
        print("LLM returned empty response for KG triple extraction.")
        return False

    print(f"Raw LLM response for KG: {raw_llm_response[:500]}...") 
    
    extracted_triples = parser_func(raw_llm_response)

    return extracted_triples

def add_triplets(current_graph, extracted_triples, base_doc_dir_for_saving):
    knowledge_graph = current_graph
    if extracted_triples:
        print(f"Extracted {len(extracted_triples)} triples.")
        new_triples_added = 0
        for s, p, o in extracted_triples:
            # Basic normalization
            s, p, o = str(s).strip(), str(p).strip(), str(o).strip()
            if s and p and o: # Ensure no empty strings
                # Add to the graph. NetworkX handles duplicate nodes/edges
                if not knowledge_graph.has_edge(s, o, key=p): 
                    knowledge_graph.add_edge(s, o, key=p, predicate_label=p) 
                    new_triples_added += 1
        if new_triples_added > 0:
            print(f"Added {new_triples_added} new unique triples to the knowledge graph.")
            print(f"KG now has {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} edges.")
            # Saving the graph
            kg_path = os.path.join(base_doc_dir_for_saving, KG_FILENAME)
            save_knowledge_graph(knowledge_graph, kg_path)
            return True
    else:
        print("No valid triples parsed from LLM response.")
    return False

def update_graph(text_content, current_graph, client, model="microsoft/phi-4", base_doc_dir_for_saving=KG_FILENAME): 
    initial_triplets = extract_triplets(text_content, client, model)
    new_triplets = filter_and_fix_triplets(current_graph, initial_triplets)
    add_triplets(current_graph, new_triplets, base_doc_dir_for_saving)

