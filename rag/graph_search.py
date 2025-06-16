import networkx as nx

def get_shortest_path_context(graph, node1, node2):
    """
    Returns the shortest path between node1 and node2 in the graph as a context string for LLMs.
    Treats all edges as undirected links. The context includes the sequence of nodes and the edge keys (relation) if available.
    """
    try:
        undirected_graph = graph.to_undirected()
        path = nx.shortest_path(undirected_graph, source=node1, target=node2)
        context_lines = []
        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]
            # Get all edges between src and tgt (in either direction)
            edges = graph.get_edge_data(src, tgt) or graph.get_edge_data(tgt, src)
            if edges:
                # Use the key as the relation
                for key, edge_attr in edges.items():
                    relation = key if key is not None else "related_to"
                    context_lines.append(f"{src} --[{relation}]-- {tgt}")
            else:
                context_lines.append(f"{src} --[related_to]-- {tgt}")
        return "\n".join(context_lines), True
    except nx.NetworkXNoPath:
        return f"No path found between '{node1}' and '{node2}'.", False
    except nx.NodeNotFound as e:
        return str(e), False

def get_all_paths_context(graph, node1, node2, max_paths=10, max_length=6):
    """
    Returns up to `max_paths` paths (as context strings) between node1 and node2 in the graph.
    Treats all edges as undirected. Each path is limited to `max_length` nodes.
    Paths are ordered from shortest to longest.
    """
    try:
        undirected_graph = graph.to_undirected()
        paths = list(nx.all_simple_paths(undirected_graph, source=node1, target=node2, cutoff=max_length))
        # Sort paths by length (shortest first)
        paths.sort(key=len)
        context_strings = []
        for idx, path in enumerate(paths):
            if idx >= max_paths:
                break
            context_lines = []
            for i in range(len(path) - 1):
                src = path[i]
                tgt = path[i + 1]
                edges = graph.get_edge_data(src, tgt) or graph.get_edge_data(tgt, src)
                if edges:
                    for key, edge_attr in edges.items():
                        relation = key if key is not None else "related_to"
                        context_lines.append(f"{src} --[{relation}]-- {tgt}")
                else:
                    context_lines.append(f"{src} --[related_to]-- {tgt}")
            context_strings.append("\n".join(context_lines))
        if not context_strings:
            return f"No paths found between '{node1}' and '{node2}'.", False
        return "\n\n---\n\n".join(context_strings), True
    except nx.NodeNotFound as e:
        return str(e), False


def get_neighborhood_subgraph(graph, start_node, hops=1):
    """
    Returns a string representation of the neighborhood subgraph (nodes and edges within N hops)
    as context for LLMs.
    Returns a tuple: (context_string, True) if successful, or (error_message, False) if the node is not found.
    """
    if not graph.has_node(start_node):
        return f"Node '{start_node}' not found in the graph.", False

    nodes_in_neighborhood = set(nx.ego_graph(graph.to_undirected(), start_node, radius=hops).nodes())
    neighborhood_subgraph = graph.subgraph(nodes_in_neighborhood)

    if not neighborhood_subgraph.nodes:
        return f"Neighborhood for '{start_node}' ({hops}-hop) is empty or the node is isolated.", False

    context_lines = [f"Neighborhood context for '{start_node}' ({hops}-hop):"]
    context_lines.append("Nodes:")
    for node in neighborhood_subgraph.nodes():
        context_lines.append(f"- {node}")

    context_lines.append("\nEdges:")
    if neighborhood_subgraph.edges:
        for u, v, data in neighborhood_subgraph.edges(data=True):
            edge_keys_data = graph.get_edge_data(u, v)
            if edge_keys_data:
                for key, attr in edge_keys_data.items():
                    relation = key if key is not None else "related_to"
                    if graph.is_directed():
                        context_lines.append(f"{u} --[{relation}]--> {v}")
                    else:
                        context_lines.append(f"{u} --[{relation}]-- {v}")
    else:
        context_lines.append("No edges in this neighborhood.")

    return "\n".join(context_lines), True

def get_common_neighbors_context(graph, node1, node2):

    """
    Finds common neighbors between node1 and node2 and returns them as a context string.
    Considers directed relationships (successors).
    Handles missing nodes gracefully.
    """
    # Check if both nodes exist in the graph
    missing = [n for n in [node1, node2] if not graph.has_node(n)]
    if missing:
        return f"Node(s) not found in the graph: {', '.join(missing)}", False

    # For MultiDiGraph, successors returns an iterator over outgoing neighbors.
    neighbors1 = set(graph.successors(node1))
    neighbors2 = set(graph.successors(node2))

    common = neighbors1.intersection(neighbors2)

    if not common:
        return f"No common outgoing neighbors found between '{node1}' and '{node2}'.", False

    context_lines = [f"Common outgoing neighbors for '{node1}' and '{node2}':"]
    for neighbor in common:
        context_lines.append(f"- {neighbor}")
        # Optionally, add relationships from node1 and node2 to the common neighbor
        edges_n1_c = graph.get_edge_data(node1, neighbor)
        if edges_n1_c:
            for key, attr in edges_n1_c.items():
                relation = key if key is not None else "related_to"
                context_lines.append(f"  {node1} --[{relation}]--> {neighbor}")
        edges_n2_c = graph.get_edge_data(node2, neighbor)
        if edges_n2_c:
            for key, attr in edges_n2_c.items():
                relation = key if key is not None else "related_to"
                context_lines.append(f"  {node2} --[{relation}]--> {neighbor}")
    return "\n".join(context_lines), True

# ======== QUESTIONS ==================

def get_most_central_location_by_degree(graph):
    """
    Identifies the most central location in the graph based on Degree Centrality.

    Args:
        graph (nx.MultiDiGraph): The input graph. Nodes are expected to have
                                 a 'ner_tag' attribute.

    Returns:
        tuple: (node_name, centrality_score) of the most central location,
               or (None, None) if no locations are found or the graph is empty.
    """
    if not graph.nodes:
        print("Graph is empty.")
        return None, None
    degree_centrality = nx.degree_centrality(graph.to_undirected())

    # Filter for nodes with the 'LOCATION' ner_tag
    location_nodes_centrality = {
        node: centrality
        for node, centrality in degree_centrality.items()
        if graph.nodes[node].get('ner_tag') == 'LOCATION'
    }

    if not location_nodes_centrality:
        print("No nodes with ner_tag 'LOCATION' found.")
        return None, None

    # Identify the location with the highest centrality score
    most_central_loc = max(location_nodes_centrality, key=location_nodes_centrality.get)
    highest_score = location_nodes_centrality[most_central_loc]

    return most_central_loc, highest_score

# most_central_loc, score = get_most_central_location_by_degree(graph)
# if most_central_loc:
#     print(f"\nQuestion: What is the most central location in the network?")
#     print(f"Centrality Measure: Degree Centrality")
#     print(f"The most central location is '{most_central_loc}' with a degree centrality score of {score:.4f}.")
# else:
#     print("Could not determine the most central location")

def get_most_influential_person_by_betweenness(graph):
    """
    Identifies the most influential individual connecting different groups or
    pieces of information based on Betweenness Centrality.

    Args:
        graph (nx.MultiDiGraph): The input graph. Nodes are expected to have
                                 a 'ner_tag' attribute.

    Returns:
        tuple: (node_name, centrality_score) of the most influential person,
               or (None, None) if no persons are found or the graph is empty.
    """
    if not graph.nodes:
        print("Graph is empty.")
        return None, None
    
    betweenness_centrality = nx.betweenness_centrality(graph.to_undirected(), normalized=True, endpoints=False)

    # Filter for nodes with the 'PERSON' ner_tag
    person_nodes_centrality = {
        node: centrality
        for node, centrality in betweenness_centrality.items()
        if graph.nodes[node].get('ner_tag') == 'PERSON'
    }

    if not person_nodes_centrality:
        print("No nodes with ner_tag 'PERSON' found.")
        return None, None

    # Identify the person with the highest centrality score
    most_influential_person = max(person_nodes_centrality, key=person_nodes_centrality.get)
    highest_score = person_nodes_centrality[most_influential_person]

    return most_influential_person, highest_score

# most_influential_p, score_p = get_most_influential_person_by_betweenness(graph)
# if most_influential_p:
#     print(f"\nQuestion: Who is the most influential individual connecting different groups or pieces of information?")
#     print(f"Centrality Measure: Betweenness Centrality")
#     print(f"The most influential person is '{most_influential_p}' with a betweenness centrality score of {score_p:.4f}.")
# else:
#     print("Could not determine the most influential person.")

def get_most_connected_person_by_degree(graph):
    """
    Identifies the person most directly connected to other entities based on Degree Centrality.

    Args:
        graph (nx.MultiDiGraph): The input graph. Nodes are expected to have
                                 a 'ner_tag' attribute.

    Returns:
        tuple: (node_name, centrality_score) of the most connected person,
               or (None, None) if no persons are found or the graph is empty.
    """
    if not graph.nodes:
        print("Graph is empty.")
        return None, None

    # Calculate Degree Centrality for all nodes
    # Using the undirected version to count all connections.
    degree_centrality = nx.degree_centrality(graph.to_undirected())

    # Filter for nodes with the 'PERSON' ner_tag
    person_nodes_centrality = {
        node: centrality
        for node, centrality in degree_centrality.items()
        if graph.nodes[node].get('ner_tag') == 'PERSON'
    }

    if not person_nodes_centrality:
        print("No nodes with ner_tag 'PERSON' found.")
        return None, None

    # Identify the person with the highest centrality score
    most_connected_person = max(person_nodes_centrality, key=person_nodes_centrality.get)
    highest_score = person_nodes_centrality[most_connected_person]

    return most_connected_person, highest_score

# most_connected_p, score_dc = get_most_connected_person_by_degree(graph)
# if most_connected_p:
#     print(f"\nQuestion: Which person is most directly connected to other entities?")
#     print(f"Centrality Measure: Degree Centrality")
#     print(f"The most directly connected person is '{most_connected_p}' with a degree centrality score of {score_dc:.4f}.")
# else:
#     print("Could not determine the most directly connected person.")

def get_most_centrally_located_entity_by_closeness(graph):
    """
    Identifies the most "centrally located" entity in the graph based on Closeness Centrality.
    This entity could be a person, location, document, event, etc.

    Args:
        graph (nx.MultiDiGraph): The input graph.

    Returns:
        tuple: (node_name, centrality_score) of the most centrally located entity,
               or (None, None) if the graph is empty.
    """
    if not graph.nodes:
        print("Graph is empty.")
        return None, None
    
    # Check if the graph has at least one edge, as closeness centrality might behave unexpectedly on totally disconnected graphs.
    if graph.number_of_edges() == 0 and graph.number_of_nodes() > 1:
        print("Graph has nodes but no edges. Closeness centrality may not be meaningful for all nodes.")
        # Fallback or specific handling for disconnected nodes if necessary
        # For now, we'll proceed, and nx will handle components.

    closeness_centrality = nx.closeness_centrality(graph.to_undirected())

    if not closeness_centrality:
        print("Could not calculate closeness centrality (e.g., graph might be empty or structured in a way that prevents calculation).")
        return None, None

    # Identify the entity with the highest centrality score
    most_central_entity = max(closeness_centrality, key=closeness_centrality.get)
    highest_score = closeness_centrality[most_central_entity]

    return most_central_entity, highest_score

# # Example usage (assuming 'graph' is your loaded knowledge graph):
# central_entity, score_cc = get_most_centrally_located_entity_by_closeness(graph)
# if central_entity:
#     print(f"\nQuestion: Which archival items or entities are most 'centrally located' for efficient topic exploration?")
#     print(f"Centrality Measure: Closeness Centrality")
#     ner_tag_central_entity = graph.nodes[central_entity].get('ner_tag', 'N/A')
#     print(f"The most centrally located entity is '{central_entity}' (Type: {ner_tag_central_entity}) with a closeness centrality score of {score_cc:.4f}.")
# else:
#     print("Could not determine the most centrally located entity.")