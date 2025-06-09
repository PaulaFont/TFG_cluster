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

def filter_edge(edge):
    """
    Checks if the number of words in the edge string is less than MAX_LEN_EDGE.
    """
    return len(edge.split()) < MAX_LEN_EDGE

def filter_triplet(triplet):
    """
    Checks if triplet needs to be filtered out because one of the phrases is in it. 
    """
    FILTER_PHRASES = [
        "el documento proporcionado",
        "no menciona",
        "no proporciona información adicional sobre",
        "la información proporcionada",
        "el documento", 
        "el contexto",
        "documento",
        "proporciona"
    ]

    for element in triplet:
        element_lower = element.lower()
        for phrase in FILTER_PHRASES:
            if phrase in element_lower:
                return False
    return True

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
  
def filter_and_fix_triplets(current_graph, initial_triplets):

    existing_triplets = current_graph.edges(keys=True) 
    existing_nodes = current_graph.nodes()

    print(f"Orignal triplets to add: {initial_triplets}")

    final_triplets_to_add = []

    for s, p, o in initial_triplets:
        # Filter out too long edges
        if not filter_edge(p):
            print(f"Skipping triplet for LONG PREDICATE: ('{s}', '{p}', '{o}')")
            continue

        # Filter out unwanted triplets using filter_triplet
        if not filter_triplet((s, p, o)):
            print(f"Skipping triplet due to filter_triplet: ('{s}', '{p}', '{o}')")
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

        # 3. After decomposition, try to match the new components to existing nodes again
        # For subject
        if sub_ner_triplets:
            # Try to match the last component of the subject chain
            last_sub_component = sub_components[-1]
            sub_match2 = find_similars(existing_nodes, last_sub_component)
            linking_subj = sub_match2 if sub_match2 else last_sub_component
        else:
            linking_subj = subj_final

        # For object
        if obj_ner_triplets:
            # Try to match the first component of the object chain
            first_obj_component = obj_components[0]
            ob_match2 = find_similars(existing_nodes, first_obj_component)
            linking_obj = ob_match2 if ob_match2 else first_obj_component
        else:
            linking_obj = obj_final

        # Add the internal links of the subject's chain if decomposed
        if sub_ner_triplets:
            final_triplets_to_add.extend(sub_ner_triplets)
        # Add the internal links of the object's chain if decomposed
        if obj_ner_triplets:
            final_triplets_to_add.extend(obj_ner_triplets)

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

