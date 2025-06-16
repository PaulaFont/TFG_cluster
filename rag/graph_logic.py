import os
import re
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from pyvis.network import Network as PyvisNetwork # Alias to avoid confusion with nx.Network
import json
from graph_utils import filter_and_fix_triplets

# --- Function to save the knowledge graph ---
def save_knowledge_graph(graph, filepath):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Knowledge graph saved to {filepath}")
    except Exception as e:
        print(f"Error saving knowledge graph: {e}")

# --- Function to load the knowledge graph ---
def load_knowledge_graph(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                graph = pickle.load(f)
            print(f"Knowledge graph loaded from {filepath}. Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
            return graph
        except Exception as e:
            print(f"Error loading knowledge graph: {e}. Starting with an empty graph.")
            return nx.MultiDiGraph()
    else:
        print("No saved knowledge graph found. Starting with an empty graph.")
        return nx.MultiDiGraph()


def produce_prompt_for_kg_extraction(input_caption):
    def _parse_to_python(string):
        triple_list_pattern = r"\[\s*(?:\(\s*(?:[^,()]+(?:,\s*[^,()]+)*)?\s*\)\s*,?\s*)+\]"
        match = re.search(triple_list_pattern, string)
        list_string = match.group(0)
        return [x for x in eval(list_string) if len(x) == 3 and isinstance(x, (tuple, list))]

    system_message = (
        "Eres un agente que recibe un fragmento de texto en el formato:\n"
        "```BOS | [contenido] | EOS```\n"
        "Tu tarea es extraer múltiples tripletas (sujeto, predicado, objeto) concisas para construir un grafo de conocimiento.\n"
        "Utiliza entidades nombradas o conceptos clave como sujetos y objetos.\n"
        "Mantén los nodos cortos (de 1 a 3 palabras), evita frases largas.\n"
        "Extrae todas las tripletas relevantes de la misma frase, usando el contexto.\n"
        "Procura que todas las tripletas esten conectadas entre si \n"
        "El texto está en español, responde siempre en español.\n"
        "Solo devuelve la lista de tripletas en este formato:\n"
        "```[('s', 'p', 'o'), ('s', 'p', 'o'), ...]```, sin explicaciones ni texto adicional."
    )

    user_message = f"Convierte esta frase en tripletas: BOS | {input_caption} | EOS. Solo devuelve la lista."

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ], _parse_to_python


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

def add_triplets(current_graph, extracted_triples, filepath, tag=False):
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
                if tag:
                    for node in [s,o]:
                        node_str = str(node)
                        if re.search(r'_\d+$', node_str):
                            knowledge_graph.nodes[node]['node_type'] = 'id_specific'
                        else:
                            knowledge_graph.nodes[node]['node_type'] = 'general'
        if new_triples_added > 0:
            print(f"Added {new_triples_added} new unique triples to the knowledge graph.")
            print(f"KG now has {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} edges.")
            # Saving the graph
            save_knowledge_graph(knowledge_graph, filepath)
            return True
    else:
        print("No valid triples parsed from LLM response.")
    return False

def update_graph(text_content, current_graph, client, filepath, model="microsoft/phi-4"): 
    initial_triplets = extract_triplets(text_content, client, model)
    new_triplets = filter_and_fix_triplets(current_graph, initial_triplets)
    add_triplets(current_graph, new_triplets, filepath) # Adds processed triplets and saves graph
    return current_graph, new_triplets

def visualize_knowledge_graph(
    graph: nx.MultiDiGraph, 
    output_format: str = "html", #html_json
    filename_prefix: str = "knowledge_graph_viz",
    output_directory: str = ".",
    show_in_browser_html: bool = True,
    notebook_mode_html: bool = False,
    html_height: str = "800px", # Increased height slightly
    html_width: str = "100%",
    default_node_color: str = "#97C2FC", # Light blue
    extra_node_color: str = "#A0A0A0", #  Grey
    default_edge_color: str = "#848484", # Grey
):
    """
    Visualizes a NetworkX knowledge graph with edge labels and coloring for HTML.
    (Args documentation remains the same, add new color args if you formalize them)
    """
    if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        print("Error: Input is not a valid NetworkX graph object.")
        return None
    
    if graph.number_of_nodes() == 0:
        print("Graph is empty. Nothing to visualize.")
        return None

    os.makedirs(output_directory, exist_ok=True)
    file_extension = "html" if "html" in output_format.lower() else output_format
    output_filepath = os.path.join(output_directory, f"{filename_prefix}.{file_extension}")

    print(f"Attempting to visualize graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Output format: {output_format.upper()}")

    try:
        if output_format == "html":
            net = PyvisNetwork(
                notebook=notebook_mode_html, 
                height=html_height, 
                width=html_width, 
                directed=isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)),
                cdn_resources='remote'
            )

            # --- Manually add nodes and edges for more control ---
            for node, node_attrs in graph.nodes(data=True):
                title = f"Node: {node}\nAttributes: {node_attrs}" # Tooltip
                node_str = str(node) # Ensure node is a string for regex fallback
                
                current_node_color = default_node_color # Default
                node_type = node_attrs.get('node_type')

                if node_type == 'id_specific':
                    current_node_color = extra_node_color
                elif node_type == 'general':
                    current_node_color = default_node_color
                else: # Fallback if 'node_type' attribute is not present or has an unexpected value
                    if re.search(r'_\d+$', node_str):
                        current_node_color = extra_node_color
                    # else it remains default_node_color (already set as default)
                
                net.add_node(node, label=str(node), title=title, color=current_node_color)

            for u, v, key, edge_attrs in graph.edges(data=True, keys=True):
                edge_label = edge_attrs.get('predicate_label', edge_attrs.get('predicate', str(key)))
                
                title = f"From: {u}\nTo: {v}\nPredicate: {edge_label}\nAttributes: {edge_attrs}" # Tooltip
                color = default_edge_color
                net.add_edge(u, v, label=str(edge_label), title=title, color=color)

            net.set_options("""
                var options = {
                    "nodes": {
                        "font": { "size": 14, "strokeWidth": 1, "strokeColor": "#ffffff" },
                        "shape": "ellipse", 
                        "size": 16,
                        "borderWidth": 1
                    },
                    "edges": {
                        "font": { "size": 10, "align": "middle", "strokeWidth": 0, "background": "rgba(255,255,255,0.7)" },
                        "color": { "inherit": false }, 
                        "arrows": {
                            "to": { "enabled": true, "scaleFactor": 0.7 }
                        },
                        "smooth": { "type": "dynamic" } 
                    },
                    "interaction": {
                        "hover": true,
                        "tooltipDelay": 200,
                        "navigationButtons": true,
                        "keyboard": true
                    },
                    "manipulation": { "enabled": false }, 
                    "physics": {
                        "enabled": true,
                        "barnesHut": {
                            "gravitationalConstant": -10000,
                            "centralGravity": 0.1,
                            "springLength": 150,
                            "springConstant": 0.05,
                            "damping": 0.09
                        },
                        "solver": "barnesHut",
                        "minVelocity": 0.75,
                        "stabilization": { "iterations": 150 }
                    }
                }
            """)

            if show_in_browser_html and not notebook_mode_html:
                net.show(output_filepath, notebook=False) 
            else:
                net.save_graph(output_filepath)
            print(f"Interactive HTML visualization saved to: {output_filepath}")
    
        elif output_format in ["png", "svg", "pdf"]:
            plt.figure(figsize=(15, 15))

            # DEFINING THE LAYOUT (depens on how dense, ...)
            try:
                pos = nx.spring_layout(graph, seed=42, k=0.8/max(1, (graph.number_of_nodes()**0.5)), iterations=50)
            except Exception as layout_err:
                print(f"Spring layout failed ({layout_err}), trying simpler layout.")
                pos = nx.circular_layout(graph) # circular_layout is deterministic and often okay

            node_colors_list = [] # Renamed to avoid conflict
            for node_item, attrs in graph.nodes(data=True):
                node_str = str(node_item)
                current_node_color = default_node_color # Default
                
                node_type = attrs.get('node_type')
                if node_type == 'id_specific':
                    current_node_color = extra_node_color
                elif node_type == 'general':
                    current_node_color = default_node_color
                else: # Fallback
                    if re.search(r'_\d+$', node_str):
                        current_node_color = extra_node_color
                node_colors_list.append(current_node_color)

            nx.draw_networkx_nodes(graph, pos, node_size=70, node_color=node_colors_list, alpha=0.9)
            nx.draw_networkx_edges(graph, pos, alpha=0.6, width=1.0, arrowsize=12, edge_color=default_edge_color)
            nx.draw_networkx_labels(graph, pos, font_size=9)
            
            if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
                edge_labels = {}
                for u, v, k_edge_key, data in graph.edges(keys=True, data=True):
                    label = data.get('predicate_label', str(k_edge_key)) 
                    edge_labels[(u, v, k_edge_key)] = label 
                if edge_labels:
                    #TODO: fix this: For now, let's create simple (u,v) labels, accepting some may overwrite.
                    simple_edge_labels = {(u,v): data.get('predicate_label', str(k_edge_key)) 
                                          for u,v,k_edge_key,data in graph.edges(keys=True, data=True)}

                    nx.draw_networkx_edge_labels(graph, pos, edge_labels=simple_edge_labels, font_size=7, 
                                                 bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.1'))

            plt.title(f"Knowledge Graph ({graph.number_of_nodes()} N, {graph.number_of_edges()} E)")
            plt.axis("off")
            plt.savefig(output_filepath, format=output_format, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Static graph visualization saved to: {output_filepath}")

        elif output_format == "gexf": 
            nx.write_gexf(graph, output_filepath)
            print(f"Graph saved in gexf format to: {output_filepath}")

        else:
            print(f"Error: Unsupported output format '{output_format}'. Supported: html, png, svg, pdf, gexf")
            return None
            
        return output_filepath

    except Exception as e:
        print(f"An error occurred during graph visualization/saving: {e}")
        import traceback
        traceback.print_exc()
        return None



