import os
import re
from openai import OpenAI
# Assuming llm_utils.py has start_llm_server and query_llm
# Assuming pre_processing.py has create_passages
from llm_utils import start_llm_server, query_llm 
from pre_processing import create_passages 
# Import all from graph_logic
from graph_logic import * 
import torch 
import json
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
import pandas as pd # create_passages might return DataFrame
import gradio as gr
import networkx as nx
import time # For cache-busting iframe

# --- Global Variables (Initialized in initialize_models_and_data) ---
passages_data = []
bi_encoder_model = None
cross_encoder_model = None
corpus_embeddings_tensor = None
llm_client_instance = None
# knowledge_graph_instance will be loaded once at startup and then managed by Gradio state per session

# --- Constants ---
# KG_FILENAME is defined in graph_logic.py, let's use it from there
# from graph_logic import KG_FILENAME 
SAVED_EMBEDDINGS_FILENAME = "corpus_embeddings.pt"
BASE_DOCUMENT_DIRECTORY = "/data/users/pfont/" # Your persistent storage for .pkl and embeddings
GRADIO_STATIC_VIZ_DIR = "./gradio_kg_visualizations/" # For HTML files served by Gradio

if not os.path.exists(BASE_DOCUMENT_DIRECTORY):
    print(f"WARNING: BASE_DOCUMENT_DIRECTORY '{BASE_DOCUMENT_DIRECTORY}' does not exist. Will attempt to create.")
    try:
        os.makedirs(BASE_DOCUMENT_DIRECTORY, exist_ok=True)
    except Exception as e:
        print(f"Could not create BASE_DOCUMENT_DIRECTORY: {e}")

if not os.path.exists(GRADIO_STATIC_VIZ_DIR):
    print(f"WARNING: GRADIO_STATIC_VIZ_DIR '{GRADIO_STATIC_VIZ_DIR}' does not exist. Will attempt to create.")
    try:
        os.makedirs(GRADIO_STATIC_VIZ_DIR, exist_ok=True)
    except Exception as e:
        print(f"Could not create GRADIO_STATIC_VIZ_DIR: {e}")


# This global will hold the graph loaded at startup
initial_knowledge_graph = nx.MultiDiGraph() 

def initialize_models_and_data(llm_model_name="microsoft/phi-4", bi_encoder_name="msmarco-bert-base-dot-v5", cross_encoder_name='cross-encoder/ms-marco-MiniLM-L6-v2', passages_filename="passages_data.json"):
    global passages_data, bi_encoder_model, cross_encoder_model, corpus_embeddings_tensor, llm_client_instance
    global initial_knowledge_graph # Use this to store the initially loaded graph

    # --- 1. Load Knowledge Graph from Disk (once at startup) ---
    kg_filepath_on_disk = os.path.join(BASE_DOCUMENT_DIRECTORY, KG_FILENAME) # KG_FILENAME from graph_logic
    initial_knowledge_graph = load_knowledge_graph(kg_filepath_on_disk) # from graph_logic.py

    # --- 2. LLM Client ---
    if not start_llm_server(llm_model_name, port=8000):
        llm_client_instance = None
        print("LLM Server failed to start.")
    else:
        llm_client_instance = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")
        print("LLM Client initialized.")
    
    # --- 3. Passages Data ---
    print("Loading passages and metadata...")
    # (Your existing logic for loading/creating passages_data from passages_filename)
    if not os.path.exists(passages_filename):
        print(f"File {passages_filename} not found. Creating new passages...")
        df_passages = create_passages() 
        passages_data = df_passages.to_dict(orient='records')
        with open(passages_filename, 'w', encoding='utf-8') as f:
            json.dump(passages_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(passages_data)} passages to {passages_filename}")
    else:
        with open(passages_filename, 'r', encoding='utf-8') as f:
            passages_data = json.load(f)
        print(f"Loaded {len(passages_data)} passages from {passages_filename}")
    if not passages_data: print("No passages. Critical error."); return
    
    # --- 4. Process Passages (Conceptual ID) ---
    # (Your existing loop to add 'conceptual_doc_id')
    for p in passages_data:
        # ... (ensure text, base_document_id, processing_version)
        p['conceptual_doc_id'] = extract_conceptual_id_from_filename(p.get('base_document_id', ''))

    passage_texts = [p['text'] for p in passages_data if 'text' in p and p['text'].strip()]
    if not passage_texts: print("No valid passage texts. Critical error."); return

    # --- 5. Embedding Models ---
    print("Loading embedding models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    try:
        bi_encoder_model = SentenceTransformer(bi_encoder_name, device=device)
        cross_encoder_model = CrossEncoder(cross_encoder_name) # device handled by input tensor
    except Exception as e: print(f"Error loading embedding models: {e}"); return
    
    # --- 6. Corpus Embeddings (Load or Generate/Save) ---
    saved_embeddings_path = os.path.join(BASE_DOCUMENT_DIRECTORY, SAVED_EMBEDDINGS_FILENAME)
    loaded_from_cache = False
    if os.path.exists(saved_embeddings_path):
        print(f"Found embeddings at {saved_embeddings_path}")
        try:
            corpus_embeddings_tensor = torch.load(saved_embeddings_path, map_location=device) 
            if corpus_embeddings_tensor.shape[0] == len(passage_texts):
                print(f"Loaded embeddings. Shape: {corpus_embeddings_tensor.shape}")
                loaded_from_cache = True
            else:
                print(f"Mismatch! Embeddings: {corpus_embeddings_tensor.shape[0]}, Texts: {len(passage_texts)}. Regenerating.")
                loaded_from_cache = False; corpus_embeddings_tensor = None
        except Exception as e:
            print(f"Error loading embeddings: {e}. Regenerating."); loaded_from_cache = False; corpus_embeddings_tensor = None

    if not loaded_from_cache:
        print("Creating embeddings...")
        corpus_embeddings_tensor = bi_encoder_model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=True)
        print(f"Embeddings created. Shape: {corpus_embeddings_tensor.shape}")
        if corpus_embeddings_tensor is not None:
            try:                    
                print(f"Saving embeddings to {saved_embeddings_path}...")
                torch.save(corpus_embeddings_tensor, saved_embeddings_path)
            except Exception as e: print(f"Error saving embeddings: {e}")

    if corpus_embeddings_tensor is None: print("Embeddings unavailable. Critical error."); return
    print("Initialization complete.")

# --- Helper functions (extract_conceptual_id, load_full_document, analyze_query, generate_answer, manage_search_terms) ---
# (These functions are as you provided in main.py - keep them here)
def extract_conceptual_id_from_filename(filename):
    if not filename: return "unknown_conceptual_doc"
    return os.path.splitext(filename)[0]

def load_full_document_by_details(filename, processing_version):
    if not filename or not processing_version: return None
    version_folder_name = f"out_llm_{processing_version}"
    filepath = os.path.join(BASE_DOCUMENT_DIRECTORY, version_folder_name, filename)
    if not os.path.exists(filepath):
        # print(f"File not found: {filepath}, trying alternatives...")
        for item in os.listdir(BASE_DOCUMENT_DIRECTORY):
            if item.startswith("out_llm_"):
                alt_path = os.path.join(BASE_DOCUMENT_DIRECTORY, item, filename)
                if os.path.exists(alt_path): filepath = alt_path; break
        else: return None # If loop completes without break
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return f.read()
    except: return None

def analyze_query_with_llm(user_query, client, model="microsoft/phi-4"): # Ensure client is passed
    # ... (your implementation)
    # For this example, I'll ensure it returns the expected dict structure
    if not client: return {"quien": [], "cuando": [], "donde": [], "que": [user_query], "consulta_refinada": user_query}
    # ... (your actual LLM call and parsing)
    return {"quien": [], "cuando": [], "donde": [], "que": [user_query], "consulta_refinada": user_query} # Placeholder for brevity


def generate_answer_with_llm(user_query, full_doc_context, client, model="microsoft/phi-4"): # Removed chunk_contexts
    # ... (your implementation, now only using full_doc_context)
    if not client: return f"LLM client not available. Mock answer for: {user_query}"
    if full_doc_context:
        return f"Answer for '{user_query}' based on doc '{full_doc_context['filename']}'."
    return f"Answer for '{user_query}' (no specific document context)."

def manage_search_terms(analyzed_query_dict):
    # ... (your implementation)
    return [analyzed_query_dict.get("consulta_refinada", "")] # Simplified


# --- Core Gradio Chat Function ---
def process_chat_turn(user_message, history, current_kg_from_state: nx.MultiDiGraph):
    """
    Handles a single turn of the chat: RAG, KG update, LLM answer, KG visualization.
    Modifies current_kg_from_state in-place.
    Returns: (chatbot_response_text, html_iframe_string_for_kg_viz)
    """
    global passages_data, bi_encoder_model, cross_encoder_model, corpus_embeddings_tensor, llm_client_instance
    
    response_parts = [] # For chatbot text

    # --- 1. RAG (Simplified for this example, use your full logic) ---
    analyzed_query = analyze_query_with_llm(user_message, llm_client_instance)
    user_query_mejorado = analyzed_query.get("consulta_refinada", user_message)
    # ... (your full retrieval and ranking to get ONE full_document_for_llm_data) ...
    # For this example, let's simulate finding one document
    # In reality, this comes from your `all_candidate_chunks_scored` and `sorted_conceptual_docs` logic
    full_document_for_llm_data = None
    # --- Simulate Retrieval to get one document ---
    # This is where your complex bi-encoder/cross-encoder logic would go
    # For now, let's assume it picks a document if query contains "document one"
    if "document one" in user_message.lower() or "doc1" in user_message.lower():
        doc_text = load_full_document_by_details("doc1.txt", "vTest") # Assuming this file exists for test
        if doc_text:
            full_document_for_llm_data = {"filename": "doc1.txt", "processing_version": "vTest", "text": doc_text}
            print(f"Retrieved document: {full_document_for_llm_data['filename']}")
    # --- End Simulate Retrieval ---

    # --- 2. Update Knowledge Graph (using the text of the single document) ---
    text_for_kg = None
    if full_document_for_llm_data:
        text_for_kg = full_document_for_llm_data['text']
    
    if text_for_kg and llm_client_instance:
        print(f"Updating KG with content from: {full_document_for_llm_data['filename'] if full_document_for_llm_data else 'N/A'}")
        # extract_triples_from_text modifies current_kg_from_state in-place
        # and saves the updated .pkl file to BASE_DOCUMENT_DIRECTORY
        extract_triples_from_text(
            text_content=text_for_kg,
            current_graph=current_kg_from_state, 
            client=llm_client_instance,
            model="microsoft/phi-4", # Or your KG extraction model
            base_doc_dir_for_saving=BASE_DOCUMENT_DIRECTORY # For .pkl file
        )
    
    # --- 3. Generate LLM Answer (using only the single full document) ---
    llm_answer = generate_answer_with_llm(
        user_query_mejorado,
        full_document_for_llm_data, # This is the dict {filename, version, text} or None
        # [], # No additional_chunks_for_llm_data
        client=llm_client_instance
    )

    # --- 4. Prepare Chatbot Text Response ---
    keywords_display_list = sorted(list(set(k for cat in ["quien","cuando","donde","que"] for k in analyzed_query.get(cat, []) if isinstance(k, str) and k.strip())))
    keywords_display_str = ", ".join(keywords_display_list) if keywords_display_list else "ninguna específica"
    response_parts.append(f"Buscando: {{{user_query_mejorado}, [{keywords_display_str}]}}")
    response_parts.append(f"\nRespuesta: {llm_answer}")
    response_parts.append("\n\n--- CONTEXTO UTILIZADO PARA GENERAR LA RESPUESTA ---")
    if full_document_for_llm_data:
        response_parts.append(f"Documento Principal: {full_document_for_llm_data['filename']} (Versión: {full_document_for_llm_data['processing_version']})")
        # Displaying full text here for transparency as per your previous request. Truncate if too long for chat.
        # response_parts.append(f"Contenido:\n{full_document_for_llm_data['text'][:1000]}...") 
    else:
        response_parts.append("No se utilizó un documento principal específico. Intenta reformular la pregunta para encontrar un documento relevante.")
    
    chatbot_response_text = "\n".join(response_parts)

    # --- 5. Generate/Update Knowledge Graph HTML Visualization ---
    kg_html_iframe_code = "<p style='text-align:center; padding:20px;'>El grafo está vacío o hubo un error.</p>" # Default
    
    # Always regenerate the HTML visualization file using the updated current_kg_from_state
    viz_filename_prefix = "current_kg_interactive_view" # Consistent name for the HTML file
    
    # The visualize_knowledge_graph function saves the HTML to GRADIO_STATIC_VIZ_DIR
    # It now returns the full absolute path to the saved HTML file
    # Edge coloring can be passed here if you have a map
    # edge_color_map = {"is_capital_of": "red", "is_in": "blue"}
    generated_html_filepath = visualize_knowledge_graph( 
        graph=current_kg_from_state, # Use the graph from the current session state
        output_format="html",
        filename_prefix=viz_filename_prefix,
        output_directory=GRADIO_STATIC_VIZ_DIR, 
        show_in_browser_html=False, # We are embedding it
        html_height="500px",
        # edge_colors_by_predicate=edge_color_map # Example
    )

    if generated_html_filepath:
        # Path for iframe src needs to be relative to how Gradio serves files
        # from the GRADIO_STATIC_VIZ_DIR.
        relative_html_filename = os.path.basename(generated_html_filepath)
        
        # Construct the servable path. Gradio's /file= endpoint needs the path relative 
        # to one of the allowed_paths. GRADIO_STATIC_VIZ_DIR itself must be an allowed_path.
        servable_file_path = os.path.join(GRADIO_STATIC_VIZ_DIR, relative_html_filename) 
        
        timestamp = int(time.time()) # Cache buster for iframe
        iframe_src = f"/file={servable_file_path}?v={timestamp}" 
        kg_html_iframe_code = f"<iframe src='{iframe_src}' width='100%' height='550px' style='border: 1px solid #ccc; min-height: 550px;'></iframe>"
        print(f"Generated iframe for Gradio with src: {iframe_src}")
    elif current_kg_from_state.number_of_nodes() == 0:
         kg_html_iframe_code = "<p style='text-align:center; padding:20px;'>Grafo de Conocimiento vacío. Realiza una consulta.</p>"

    return chatbot_response_text, kg_html_iframe_code


# --- Gradio Interface Setup ---
with gr.Blocks(theme="default") as demo:
    # Chat state to store the knowledge graph across interactions for this session
    # Initialize with a copy of the globally loaded graph, or a new one if global is None
    initial_kg_for_session = initial_knowledge_graph.copy() if initial_knowledge_graph is not None else nx.MultiDiGraph()
    chat_state = gr.State(value=initial_kg_for_session) 

    gr.Markdown("# Chatbot Histórico con Grafo de Conocimiento Dinámico")
    with gr.Row():
        with gr.Column(scale=6): # Chatbot column
            chatbot_display = gr.Chatbot(label="Chatbot", height=600, show_label=False)
            input_message_box = gr.Textbox(label="Tu Pregunta:", placeholder="Escribe algo y presiona Enter...", lines=2, show_label=False)
        with gr.Column(scale=4): # Knowledge Graph column
            gr.Markdown("### Grafo de Conocimiento")
            kg_html_viewer = gr.HTML(
                # Initial content for the HTML viewer
                value="<p style='text-align:center; padding:20px;'>El grafo se mostrará aquí.</p>", 
                label="Visualización del Grafo" # Label might not be visible depending on theme/layout
            )

    clear_button = gr.Button("🧹 Limpiar Chat y Grafo de Sesión")

    def handle_user_message(user_msg, chat_history_list, current_session_kg):
        # Append user message to history for display
        chat_history_list.append((user_msg, None)) # Use tuple for gr.Chatbot
        # The bot's response will be filled in by the next step
        return "", chat_history_list, current_session_kg # Clear textbox, pass updated history & state

    def get_bot_response(chat_history_list, current_session_kg):
        user_msg = chat_history_list[-1][0] # Get the latest user message
        
        # Call the main processing function
        # current_session_kg is modified in-place by process_chat_turn if extract_triples works on the passed graph
        bot_text_response, kg_html_output = process_chat_turn(user_msg, chat_history_list, current_session_kg)
        
        chat_history_list[-1] = (user_msg, bot_text_response) # Update history with bot's response
        
        return chat_history_list, kg_html_output, current_session_kg # Return updated history, new KG HTML, and updated KG state

    input_message_box.submit(
        handle_user_message, 
        [input_message_box, chatbot_display, chat_state], 
        [input_message_box, chatbot_display, chat_state] # Outputs for user_fn
    ).then(
        get_bot_response, 
        [chatbot_display, chat_state], # Inputs for bot_fn
        [chatbot_display, kg_html_viewer, chat_state] # Outputs for bot_fn: update chatbot, kg_html_viewer, and state
    )

    def clear_session_chat_and_kg():
        # Resets the session's graph to a new empty one. Does not delete the .pkl file.
        new_empty_kg_for_session = nx.MultiDiGraph()
        
        # Generate HTML for an empty graph to clear the viewer
        empty_viz_path = visualize_knowledge_graph( # From graph_logic
            graph=new_empty_kg_for_session, 
            output_directory=GRADIO_STATIC_VIZ_DIR,
            filename_prefix="current_kg_interactive_view", # Keep filename consistent
            show_in_browser_html=False
        )
        empty_iframe_html = "<p style='text-align:center; padding:20px;'>Grafo de sesión limpiado. Realiza una consulta.</p>"
        if empty_viz_path:
            relative_empty_filename = os.path.basename(empty_viz_path)
            servable_empty_path = os.path.join(GRADIO_STATIC_VIZ_DIR, relative_empty_filename)
            # Add timestamp to force iframe reload of the (now empty) graph file
            empty_iframe_src = f"/file={servable_empty_path}?v={int(time.time())}"
            empty_iframe_html = f"<iframe src='{empty_iframe_src}' width='100%' height='550px' style='border: 1px solid #ccc;  min-height: 550px;'></iframe>"
            
        return [], empty_iframe_html, new_empty_kg_for_session

    clear_button.click(clear_session_chat_and_kg, None, [chatbot_display, kg_html_viewer, chat_state])


if __name__ == "__main__":
    # This runs once when the script starts
    initialize_models_and_data()
    print(f"Global 'initial_knowledge_graph' loaded with {initial_knowledge_graph.number_of_nodes()} nodes.")

    # The Gradio app will use a *copy* of this for each session's state,
    # or you can make chat_state directly use the global one if you want all sessions to share one live graph.
    # For per-session graphs that build on the initial load:
    # (initial_kg_for_session is already set up like this inside gr.Blocks context)

    print(f"Gradio will serve files from: {os.path.abspath(GRADIO_STATIC_VIZ_DIR)}")
    print(f"Persistent KG (.pkl) is in: {os.path.abspath(BASE_DOCUMENT_DIRECTORY)}")
    
    demo.queue().launch(allowed_paths=[GRADIO_STATIC_VIZ_DIR, BASE_DOCUMENT_DIRECTORY])
    # share=True for public link if needed