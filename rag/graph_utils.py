import os
import re
import networkx as nx
import pickle
import spacy
import Levenshtein as levenshtein
from num2words import num2words
import math
from ner_logic import ner_function, link_components_by_context

MAX_LEN_NODE = 6 #(words)
MAX_LEN_EDGE = 5 #(words)
GRAPH_DIRECTORY = "/data/users/pfont/graph/"
KG_FILENAME = "online_knowledge_graph_tests.pkl" 

def filter_edge(edge):
    """
    Checks if the number of words in the edge string is less than MAX_LEN_EDGE.
    """
    return len(edge.split()) < MAX_LEN_EDGE



# TODO: Apply NER in here. (Future idea: to provide information to the node as, the type: PER, LOC, ORG, DATE, ...)
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

# ------------------------------------------------------------------------
# Apply string edit distance
# ------------------------------------------------------------------------

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

def find_similars(lst, word):
    """
    Finds the first string in lst that is similar to word using are_strings_similar.
    If more than one match is found, prints the first match.
    Returns the first similar string, or None if no match is found.
    """
    matches = [item for item in lst if are_strings_similar(item, word)]
    if matches:
        if len(matches) > 1:
            print(f"Multiple similar strings found. Returning the first: {matches[0]}")
        return matches[0]
    return None
  

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


def filter_and_fix_triplets(current_graph, triplets):
    #TODO: remove, now it's temporary
    existing_triplets = [
        # About the Person (Subject)
        ('Francisco Iacasta Catalán', 'labriego', 'era'),
        ('Francisco Iacasta Catalán', '46 años', 'tenía'),
        ('Francisco Iacasta Catalán', 'Olite', 'era_natural_de'),
        ('Francisco Iacasta Catalán', 'Olite', 'era_vecino_de'),

        # The Trial and Accusation
        ('Francisco Iacasta Catalán', 'Consejo de Guerra', 'fue_juzgado_por'),
        ('Consejo de Guerra', 'Pamplona', 'tuvo_lugar_en'),
        ('Consejo de Guerra', '22 de enero de 1943', 'fecha_de_juicio'),
        ('Francisco Iacasta Catalán', 'auxilio a la Rebelión', 'fue_acusado_de'),

        # The Court's Composition
        ('Consejo de Guerra', 'Rodrigo Torrent', 'fue_presidido_por'),
        ('Rodrigo Torrent', 'Teniente Coronel', 'tenía_título_de'),
        ('Fallo', 'vocales del consejo', 'fue_confirmado_por'),

        # Requests from Prosecution and Defense
        ('Ministerio Fiscal', 'reclusión menor de dos a nueve años', 'solicitó_pena'),
        ('Defensa', 'absolución o pena menor', 'solicitó'),

        # The Verdict and Sentence
        ('Consejo de Guerra', 'Francisco Iacasta Catalán', 'condenó'),
        ('Consejo de Guerra', 'dos años y seis meses de reclusión menor', 'impuso_pena'),
        ('Condena', 'artículo 240 del Código de Justicia Militar', 'basada_en'),

        # Dissenting Opinion
        ('Voto particular', 'un año de reclusión menor', 'propuso_pena')
    ]
    existing_nodes = list(set([t[0] for t in existing_triplets] + [t[1] for t in existing_triplets]))

    triplets = current_graph.edges(keys=True) #node, node, edge #TODO: fix later
    print(f"Orignal triplets to add: {triplets}")
    current_triplets = []
    for s, o, e in triplets:  # TODO: check order that triplets come from the LLM
        if filter_edge(e):  # Short edge
            sub_match = find_similars(existing_nodes, s)
            ob_match = find_similars(existing_nodes, o)

            subj_final = sub_match if sub_match else s
            obj_final = ob_match if ob_match else o

            if sub_match:
                print(f"Matched subject '{s}' to existing node '{sub_match}'")
            if ob_match:
                print(f"Matched object '{o}' to existing node '{ob_match}'")

            current_triplets.append((subj_final, e, obj_final))
            
            temporary_new_triplets = []
            # NER (new triplets)
            if not sub_match:
                sub_list = ner_function(s)
                if sub_list and sub_list != [s]:
                    print(f"NER for subject '{s}': {sub_list}")
                    new_triplets_sub = link_components_by_context(s, sub_list)
                    temporary_new_triplets += new_triplets_sub
            if not ob_match:
                ob_list = ner_function(o)
                if ob_list and ob_list != [o]:
                    print(f"NER for object '{o}': {ob_list}")
                    new_triplets_ob = link_components_by_context(o, ob_list)         
                    temporary_new_triplets += new_triplets_ob
            
            # Check if the new nodes already exist:
            for s, o, e in temporary_new_triplets: 
                sub_match = find_similars(existing_nodes, s)
                ob_match = find_similars(existing_nodes, o)

                subj_final = sub_match if sub_match else s
                obj_final = ob_match if ob_match else o

                if sub_match:
                    print(f"Matched subject '{s}' to existing node '{sub_match}'")
                if ob_match:
                    print(f"Matched object '{o}' to existing node '{ob_match}'")

                current_triplets.append((subj_final, e, obj_final))

        else: 
            print(f"Removed triplet for LONG EDGE ({s} {e} {o})")
    print(f"New triplets to add: {current_triplets}")
    return current_triplets


def filter_and_fix_triplets2(current_graph, initial_triplets):
     #TODO: remove, now it's temporary
    existing_triplets = [
        # About the Person (Subject)
        ('Francisco Iacasta Catalán', 'labriego', 'era'),
        ('Francisco Iacasta Catalán', '46 años', 'tenía'),
        ('Francisco Iacasta Catalán', 'Olite', 'era_natural_de'),
        ('Francisco Iacasta Catalán', 'Olite', 'era_vecino_de'),

        # The Trial and Accusation
        ('Francisco Iacasta Catalán', 'Consejo de Guerra', 'fue_juzgado_por'),
        ('Consejo de Guerra', 'Pamplona', 'tuvo_lugar_en'),
        ('Consejo de Guerra', '22 de enero de 1943', 'fecha_de_juicio'),
        ('Francisco Iacasta Catalán', 'auxilio a la Rebelión', 'fue_acusado_de'),

        # The Court's Composition
        ('Consejo de Guerra', 'Rodrigo Torrent', 'fue_presidido_por'),
        ('Rodrigo Torrent', 'Teniente Coronel', 'tenía_título_de'),
        ('Fallo', 'vocales del consejo', 'fue_confirmado_por'),

        # Requests from Prosecution and Defense
        ('Ministerio Fiscal', 'reclusión menor de dos a nueve años', 'solicitó_pena'),
        ('Defensa', 'absolución o pena menor', 'solicitó'),

        # The Verdict and Sentence
        ('Consejo de Guerra', 'Francisco Iacasta Catalán', 'condenó'),
        ('Consejo de Guerra', 'dos años y seis meses de reclusión menor', 'impuso_pena'),
        ('Condena', 'artículo 240 del Código de Justicia Militar', 'basada_en'),

        # Dissenting Opinion
        ('Voto particular', 'un año de reclusión menor', 'propuso_pena')
    ]
    existing_nodes = list(set([t[0] for t in existing_triplets] + [t[1] for t in existing_triplets]))

    initial_triplets = current_graph.edges(keys=True) #node, node, edge #TODO: fix later
    print(f"Orignal triplets to add: {initial_triplets}")
    final_triplets_to_add = []

    for s, o, p in initial_triplets:
        if not filter_edge(p):
            print(f"SKIPPING triplet for LONG PREDICATE: ('{s}', '{p}', '{o}')")
            continue
            
        # 1. Match nodes to existing ones in the graph
        sub_match = find_similars(existing_nodes, s)
        ob_match = find_similars(existing_nodes, o)

        subj_final = sub_match if sub_match else s
        obj_final = ob_match if ob_match else o
        
        # Log matches
        if sub_match: print(f"Matched subject '{s}' -> '{subj_final}'")
        if ob_match: print(f"Matched object '{o}' -> '{obj_final}'")

        # 2. Decompose new nodes and prepare internal links
        sub_ner_triplets = []
        obj_ner_triplets = []
        
        # Decompose subject only if it's a new node
        if not sub_match and len(s.split()) > MAX_LEN_NODE:
            sub_components = ner_function(s)
            if len(sub_components) > 1:
                print(f"NER decomposed subject '{s}': {sub_components}")
                sub_ner_triplets = link_components_by_context(s, sub_components)
        else:
            sub_components = [subj_final]
            
        # Decompose object only if it's a new node
        if not ob_match and len(o.split()) > MAX_LEN_NODE:
            obj_components = ner_function(o)
            if len(obj_components) > 1:
                print(f"NER decomposed object '{o}': {obj_components}")
                obj_ner_triplets = link_components_by_context(o, obj_components)
        else:
            obj_components = [obj_final]
            
        # 3. Build the final list of triplets based on decompositions
        linking_subj = subj_final
        linking_obj = obj_final

        # If the subject was decomposed...
        if sub_ner_triplets:
            # Add the internal links of the subject's chain.
            final_triplets_to_add.extend(sub_ner_triplets)
            # The new "head" of the main relationship is the LAST node of the subject's chain.
            linking_subj = sub_components[-1]

        # If the object was decomposed...
        if obj_ner_triplets:
            # Add the internal links of the object's chain.
            final_triplets_to_add.extend(obj_ner_triplets)
            # The new "tail" of the main relationship is the FIRST node of the object's chain.
            linking_obj = obj_components[0]
            
        # Add the main, re-linked triplet. This connects the subject (or its chain's end)
        # to the object (or its chain's start).
        main_triplet = (linking_subj, p, linking_obj)
        print(f"Adding main link: {main_triplet}")
        final_triplets_to_add.append(main_triplet)
        
        #Remove all triplets with nodes longer than MAX_LEN_NODE
        filtered_triplets = []
        for triplet in final_triplets_to_add:
            subj_len = len(triplet[0].split())
            obj_len = len(triplet[2].split())
            if subj_len > MAX_LEN_NODE or obj_len > MAX_LEN_NODE:
                print(f"Skipping triplet due to long node: {triplet}")
            else:
                filtered_triplets.append(triplet)
        final_triplets_to_add = filtered_triplets

    print(f"\nFinal, processed triplets to add to graph: {final_triplets_to_add}")
    return final_triplets_to_add
# ------------------------------------------------------------------------
# LLM extraction and graph update
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


filepath = os.path.join(GRAPH_DIRECTORY, KG_FILENAME)
if os.path.exists(filepath):
    try:
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
        print(f"Knowledge graph loaded from {filepath}. Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
    except Exception as e:
        print(f"Error loading knowledge graph: {e}. Starting with an empty graph.")
        
filter_and_fix_triplets2(graph, [])