from graph_utils import find_similars
from graph_logic import add_triplets
from ner_logic import filter_nodes_global
import re 

def add_tag(text, document_id):
    return f"{text}_{str(document_id)}"

def update_triplets_id(triplets, document_id):
    nodes = set(node for s, _, o in triplets for node in (s, o))
    references = {}
    attributes = {}

    for item in nodes:
        is_global, attribute = filter_nodes_global(item)
        if is_global:
            references[item] = item
            attributes[item] = attribute
        else:
            references[item] = add_tag(item, document_id)
            attributes[add_tag(item, document_id)] = attribute

    new_triplets = []
    for s, p, o in triplets:
        new_triplets.append((references[s], p, references[o]))

    return new_triplets, attributes


def get_triplets_to_add(global_nodes, triplets, document_id):
    new_triplets, attributes = update_triplets_id(triplets, document_id)

    # We add global nodes that don't have an ID
    existing_global_nodes = set()
    for item in global_nodes:
        if not re.search(r'_\d+$', item):  # Check if item is a global node
            existing_global_nodes.add(item)
    existing_global_nodes_list = list(existing_global_nodes)

    final_triplets = []

    for s,p,o in new_triplets:
        current_s = s
        current_o = o
        if not re.search(r'_\d+$', s):
            # Try to find a similar node among the existing global nodes
            s_match = find_similars(existing_global_nodes_list, s)
            if s_match:
                current_s = s_match
                attributes[s_match] = attributes[current_s]

        if not re.search(r'_\d+$', o):
            # Try to find a similar node among the existing global nodes
            o_match = find_similars(existing_global_nodes_list, o)
            if o_match:
                current_o = o_match
                attributes[o_match] = attributes[current_o]
        
        final_triplets.append((current_s, p, current_o))
    return final_triplets, attributes


def update_global_graph(processed_triplets, global_graph, document_id, filepath): 
    existing_nodes = global_graph.nodes()
    triplets_to_add, attributes = get_triplets_to_add(existing_nodes, processed_triplets, document_id)
    add_triplets(global_graph, triplets_to_add, filepath, tag=True, attributes=attributes) # Adds processed triplets and saves graph
    return global_graph