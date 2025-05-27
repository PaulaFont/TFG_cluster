import os
from openai import OpenAI 
import networkx as nx
from ../llm_utils import * 

from ../graph_logic import (
    visualize_knowledge_graph,
    load_knowledge_graph,
    extract_triples_from_text,
    KG_FILENAME 
)

LLM_MODEL_FOR_EXTRACTION = "microsoft/phi-4"
DEMO_BASE_DIRECTORY = "./graph_demo_files/"  
llm_client = None

if not start_llm_server(LLM_MODEL_FOR_EXTRACTION, port=8000):
        llm_client = None
        print("LLM Server failed to start. LLM functionalities will be unavailable.")
else:
    llm_client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123" 
    )
    print("LLM Client initialized.")

def run_demo(add_edges=True):
    print("--- Knowledge Graph Online Creation Demo ---")

    # 1. Setup: Create demo directory and ensure it's clean
    if os.path.exists(DEMO_BASE_DIRECTORY):
        pass 
    os.makedirs(DEMO_BASE_DIRECTORY, exist_ok=True)
    
    kg_filepath = os.path.join(DEMO_BASE_DIRECTORY, KG_FILENAME)

    # 2. Initial Load (Graph should be empty or loaded from previous demo run)
    print("\n--- Initializing/Loading Knowledge Graph ---")
    current_kg_instance = load_knowledge_graph(kg_filepath)
    print(f"Initial KG: {len(current_kg_instance.nodes)} nodes, {len(current_kg_instance.edges)} edges")
    if len(current_kg_instance.nodes):
        print("Nodes:", list(current_kg_instance.nodes(data=True))[:5]) # Print first 5 nodes
        print("Edges:", list(current_kg_instance.edges(data=True, keys=True))[:5]) # Print first 5 edges

    if (add_edges):
        # 3. Simulate User Queries and KG Updates
        sample_texts = [
            "Paris is the capital of France. France is in Europe.",
            "The Eiffel Tower is a famous landmark in Paris, built by Gustave Eiffel.",
            "Berlin is the capital of Germany. Germany is also in Europe.",
            "PRESIDENTE Coronel de Artillería D. Manuel Suárez Sánchez"
        ]
        sample_texts = ["Durante la guerra, Don Pedro Manuel Ruiz Pardo, un funcionario del Cuerpo Pericial de Aduanas y vecino de Errezil, fue acusado de haber iniciado el Movimiento Nacional en la Aduana de Errezil y de cooperar con el Gobierno marxista para facilitar el paso de elementos de derechas. Un Consejo de Guerra ordinario de plaza en Pamplona lo condenó inicialmente a dos años y un día de prisión temporal y veinte años de reclusión menor por un delito de auxilio a la rebelión. Sin embargo, el Tribunal Supremo de Justicia Militar revocó esta sentencia y lo condenó a doce años y un día de reclusión menor, conmutada posteriormente por siete años de prisión menor. Su hijo murió luchando en el Ejército Nacional."]
        
        if not llm_client:
            print("\nWARNING: LLM Client not available. Skipping live triple extraction.")
            print("To run live extraction, ensure OpenAI client is configured or provide a mock.")
            current_kg_instance.add_edge("MockSubject", "MockObject", key="is_mocked", predicate_label="is_mocked")
            print("Added mock triple to KG.")
        else:
            for i, text in enumerate(sample_texts):
                print(f"\n--- Processing Text {i+1} ---")
                extract_triples_from_text(
                    text_content=text,
                    current_graph=current_kg_instance,
                    client=llm_client,
                    model=LLM_MODEL_FOR_EXTRACTION,
                    base_doc_dir_for_saving=DEMO_BASE_DIRECTORY
                )

    # 4. Inspect the Final Graph (In-memory)
    print("\n--- Final Knowledge Graph State (In-Memory) ---")
    print(f"Total Nodes: {len(current_kg_instance.nodes)}")
    print(f"Total Edges: {len(current_kg_instance.edges)}")
    
    if len(current_kg_instance.nodes):
        print("\nSample Nodes (first 10):")
        for node in list(current_kg_instance.nodes)[:10]:
            print(f"- {node}")

    if len(current_kg_instance.edges):
        print("\nSample Edges (first 10 with predicates):")
        for u, v, key, data in list(current_kg_instance.edges(data=True, keys=True))[:10]:
            print(f"- ('{u}', '{data.get('predicate_label', key)}', '{v}')") # Use predicate_label if available
    
    if current_kg_instance.number_of_nodes() > 0 :
        print("\n--- Visualizing the Graph ---")

        # Option 1: Save as an interactive HTML file
        html_file_path = visualize_knowledge_graph(
            graph=current_kg_instance,
            output_format="html",
            filename_prefix="my_interactive_kg",
            output_directory=DEMO_BASE_DIRECTORY, 
            show_in_browser_html=False,
            default_node_color="#A0A0A0",
            default_edge_color="#C0C0C0", 
        )

        if html_file_path:
            print(f"Generated HTML at: {html_file_path}")

       # Option 2: Save as a static PNG image
        png_file_path = visualize_knowledge_graph(
            graph=current_kg_instance,
            output_format="png",
            filename_prefix="my_static_kg",
            output_directory=DEMO_BASE_DIRECTORY
        )
        if png_file_path:
            print(f"Generated PNG at: {png_file_path}")

        # Option 3: Save as GEXF
        gml_file_path = visualize_knowledge_graph(
            graph=current_kg_instance,
            output_format="gexf",
            filename_prefix="my_kg_data",
            output_directory=DEMO_BASE_DIRECTORY
        )
        if gml_file_path:
            print(f"Generated GML at: {gml_file_path} (Can be opened in Gephi, Cytoscape, etc.)")
    else:
        print("Graph is empty, skipping visualization.")

    print("\n--- Demo Complete ---")
    print(f"The knowledge graph is saved/updated at: {kg_filepath}")

if __name__ == "__main__":
    run_demo()