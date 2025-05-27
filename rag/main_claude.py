# rag_system.py
import os
import re
from openai import OpenAI
from llm_utils import * 
from pre_processing import *
from graph_logic import * 
import torch 
import json
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
import pandas as pd
import gradio as gr
import networkx as nx

class RAGSystem:
    """
    A class to encapsulate all RAG system state and functionality.
    This eliminates the need for global variables.
    """
    
    def __init__(self, base_document_directory="/data/users/pfont/", 
                 graph_document_directory=None):
        # Configuration
        self.BASE_DOCUMENT_DIRECTORY = base_document_directory
        self.GRAPH_DOCUMENT_DIRECTORY = graph_document_directory or os.path.join(base_document_directory, "graph/")
        self.SAVED_EMBEDDINGS_FILENAME = "corpus_embeddings.pt"
        self.LLM_MODEL_NAME = "microsoft/phi-4"
        self.GRAPH_FILENAME = "online_knowledge_graph"
        
        # Create directories
        os.makedirs(self.GRAPH_DOCUMENT_DIRECTORY, exist_ok=True)
        if not os.path.exists(self.BASE_DOCUMENT_DIRECTORY):
            print(f"WARNING: BASE_DOCUMENT_DIRECTORY '{self.BASE_DOCUMENT_DIRECTORY}' does not exist.")
        
        # State variables (previously globals)
        self.passages_data = []
        self.bi_encoder_model = None
        self.cross_encoder_model = None
        self.corpus_embeddings_tensor = None
        self.llm_client_instance = None
        self.knowledge_graph_instance = nx.MultiDiGraph()
        self.html_file_path = "/data/users/pfont/graph/online_knowledge_graph.html"
        
        # Search configuration
        self.MAX_SEARCH_TERMS = 5
        self.BI_ENCODER_TOP_K = 20
        self.CROSS_ENCODER_THRESHOLD = 0.0
        self.MAX_ADDITIONAL_CHUNKS_FOR_LLM = 0

    def initialize_models_and_data(self, 
                                 bi_encoder_name="msmarco-bert-base-dot-v5", 
                                 cross_encoder_name='cross-encoder/ms-marco-MiniLM-L6-v2', 
                                 passages_filename="passages_data.json"):
        """Initialize all models and data"""
        
        # GRAPH
        kg_filepath = os.path.join(self.GRAPH_DOCUMENT_DIRECTORY, self.GRAPH_FILENAME + ".pkl")
        self.knowledge_graph_instance = load_knowledge_graph(kg_filepath)

        # LLM
        if not start_llm_server(self.LLM_MODEL_NAME, port=8000):
            self.llm_client_instance = None
            print("LLM Server failed to start. LLM functionalities will be unavailable.")
        else:
            self.llm_client_instance = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="token-abc123" 
            )
            print("LLM Client initialized.")
        
        # TEXT CHUNKS
        self._load_or_create_passages(passages_filename)
        
        # EMBEDDING MODELS
        self._initialize_embedding_models(bi_encoder_name, cross_encoder_name)
        
        # CORPUS EMBEDDINGS
        self._load_or_create_embeddings()
        
        print("Initialization complete.")

    def _load_or_create_passages(self, passages_filename):
        """Load or create passage data"""
        print("Creating/Loading passages and metadata...")
        if not os.path.exists(passages_filename):
            print(f"File {passages_filename} not found. Creating new passages...")
            df_passages = create_passages() 
            self.passages_data = df_passages.to_dict(orient='records')
            with open(passages_filename, 'w', encoding='utf-8') as f:
                json.dump(self.passages_data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.passages_data)} passages to {passages_filename}")
        else:
            with open(passages_filename, 'r', encoding='utf-8') as f:
                self.passages_data = json.load(f)
            print(f"Loaded {len(self.passages_data)} passages from {passages_filename}")

        if not self.passages_data:
            raise ValueError("No passages were loaded or created. Critical error.")

    def _initialize_embedding_models(self, bi_encoder_name, cross_encoder_name):
        """Initialize embedding models"""
        print("Loading embedding models...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        try:
            self.bi_encoder_model = SentenceTransformer(bi_encoder_name, device=device)
            self.cross_encoder_model = CrossEncoder(cross_encoder_name, device=device) 
        except Exception as e:
            raise RuntimeError(f"Error loading sentence-transformer models: {e}")

    def _load_or_create_embeddings(self):
        """Load or create corpus embeddings"""
        passage_texts = [p['text'] for p in self.passages_data if 'text' in p and p['text'].strip()]
        if not passage_texts:
            raise ValueError("Passages data contains no valid text entries.")

        saved_embeddings_path = os.path.join(self.BASE_DOCUMENT_DIRECTORY, self.SAVED_EMBEDDINGS_FILENAME)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        loaded_from_cache = False
        if os.path.exists(saved_embeddings_path):
            print(f"Found pre-computed embeddings at {saved_embeddings_path}")
            try:
                print("Loading corpus embeddings...")
                self.corpus_embeddings_tensor = torch.load(saved_embeddings_path, map_location=device) 
                if self.corpus_embeddings_tensor.shape[0] == len(passage_texts):
                    print(f"Loaded corpus embeddings. Shape: {self.corpus_embeddings_tensor.shape}")
                    loaded_from_cache = True
                else:
                    print(f"Mismatch! Will regenerate embeddings.")
                    self.corpus_embeddings_tensor = None
                    loaded_from_cache = False
            except Exception as e:
                print(f"Error loading from cache: {e}. Will regenerate.")
                loaded_from_cache = False
                self.corpus_embeddings_tensor = None

        if not loaded_from_cache:
            print("Creating embeddings for all passages...")
            self.corpus_embeddings_tensor = self.bi_encoder_model.encode(
                passage_texts, convert_to_tensor=True, show_progress_bar=True
            )
            print(f"Corpus embeddings created. Shape: {self.corpus_embeddings_tensor.shape}")
            try:                    
                print(f"Saving new corpus embeddings to {saved_embeddings_path}...")
                torch.save(self.corpus_embeddings_tensor, saved_embeddings_path)
            except Exception as e:
                print(f"Error saving new corpus embeddings: {e}")

    def load_full_document_by_details(self, filename, processing_version):
        """Load the full text content of a document"""
        if not filename or not processing_version:
            print("Error: Filename or processing_version missing.")
            return None

        version_folder_name = f"out_llm_{processing_version}"
        filepath = os.path.join(self.BASE_DOCUMENT_DIRECTORY, version_folder_name, filename)

        if not os.path.exists(filepath):
            print(f"Error: Document file '{filepath}' not found.")
            # Try alternative folders
            for item in os.listdir(self.BASE_DOCUMENT_DIRECTORY):
                if item.startswith("out_llm_"):
                    alt_path = os.path.join(self.BASE_DOCUMENT_DIRECTORY, item, filename)
                    if os.path.exists(alt_path):
                        print(f"Found '{filename}' in alternative folder: {item}")
                        filepath = alt_path
                        break
            else:
                return None
                
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading document file '{filepath}': {e}")
            return None

    def analyze_query_with_llm(self, user_query):
        """Analyze query with LLM to extract entities"""
        prompt = f"""
        Analiza la siguiente consulta en español e identifica las entidades clave.

        Consulta del usuario: "{user_query}"

        Extrae y clasifica las siguientes entidades:
        1. QUIÉN (Personas): Nombres de personas, grupos, organizaciones, etc.
        2. CUÁNDO (Tiempo): Fechas, períodos, siglos, años, meses, días, etc.
        3. DÓNDE (Lugar): Ubicaciones geográficas, países, ciudades, regiones, etc.
        4. QUÉ (Tema): El tema principal, eventos, conceptos, términos específicos, etc.

        Responde en el siguiente formato JSON:
        {{
        "quien": ["entidad1", "entidad2", ...] o [],
        "cuando": ["entidad1", "entidad2", ...] o [],
        "donde": ["entidad1", "entidad2", ...] o [],
        "que": ["entidad1", "entidad2", ...] o [],
        "consulta_refinada": "Una consulta concisa optimizada para búsqueda semántica"
        }}

        Solo devuelve el JSON, sin comentarios ni explicaciones adicionales.
        """
        
        if self.llm_client_instance:
            response_text = query_llm(self.llm_client_instance, self.LLM_MODEL_NAME, prompt)
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}')
                if json_start != -1 and json_end != -1 and json_start < json_end:
                    json_str = response_text[json_start:json_end+1]
                    result = json.loads(json_str)
                    for key in ["quien", "cuando", "donde", "que"]:
                        if key not in result or not isinstance(result[key], list): 
                            result[key] = []
                    if "consulta_refinada" not in result or not isinstance(result["consulta_refinada"], str):
                        result["consulta_refinada"] = user_query
                    return result
                else: 
                    raise json.JSONDecodeError("No JSON object found", response_text, 0)
            except json.JSONDecodeError as e:
                print(f"LLM JSON parse error: {e}")
                return {"quien": [], "cuando": [], "donde": [], "que": [user_query], "consulta_refinada": user_query}
        else:
            return {"quien": [], "cuando": [], "donde": [], "que": [user_query], "consulta_refinada": user_query}

    def generate_answer_with_llm(self, user_query, full_doc_context):
        """Generate answer using LLM with provided context"""
        contexts_for_prompt = []
        if full_doc_context and full_doc_context.get('text'):
            contexts_for_prompt.append(
                f"--- DOCUMENTO COMPLETO PRINCIPAL (Archivo: {full_doc_context.get('filename', 'Desconocido')}, Versión de procesamiento: {full_doc_context.get('processing_version', 'N/A')}) ---\n"
                f"{full_doc_context['text']}\n"
                f"--- FIN DEL DOCUMENTO COMPLETO PRINCIPAL ---"
            )
        
        if not contexts_for_prompt:
            return "No he encontrado información relevante en los documentos para responder a tu pregunta."

        context_str = "\n\n".join(contexts_for_prompt)
        
        prompt = f"""
        Eres un asistente de IA experto en documentos históricos en español.
        Responde a la pregunta del usuario basándote *únicamente* en el "DOCUMENTO COMPLETO PRINCIPAL" y, si es necesario, en los "FRAGMENTOS ADICIONALES" proporcionados.
        Prioriza la información del "DOCUMENTO COMPLETO PRINCIPAL". Usa los fragmentos adicionales para complementar si es necesario.
        Sé conciso y responde directamente. Sintetiza la información. No inventes nada.
        Si la información no está en los textos, indícalo.
        No menciones "DOCUMENTO COMPLETO PRINCIPAL" o "FRAGMENTOS ADICIONALES" en tu respuesta final.

        Pregunta del Usuario: "{user_query}"

        Contextos Proporcionados:
        {context_str}

        Respuesta Concisa:
        """
        if self.llm_client_instance:
            return query_llm(self.llm_client_instance, self.LLM_MODEL_NAME, prompt)
        else:
            return f"Respuesta de muestra para '{user_query}' basada en {'documento completo' if full_doc_context else ''}"

    def manage_search_terms(self, analyzed_query_dict):
        """Extract and manage search terms from analyzed query"""
        search_terms = []
        refined_query = analyzed_query_dict.get("consulta_refinada")
        if refined_query and isinstance(refined_query, str) and refined_query.strip(): 
            search_terms.append(refined_query.strip())
        
        for key_type in ["quien", "cuando", "donde", "que"]:
            keywords = analyzed_query_dict.get(key_type, [])
            if isinstance(keywords, list): 
                search_terms.extend(k for k in keywords if isinstance(k, str) and k.strip())
            elif isinstance(keywords, str) and keywords.strip(): 
                search_terms.append(keywords.strip())
        
        return sorted(list(set(search_terms))) if search_terms else []

    def chat_search(self, message, history):
        html_path = "hi"
        """Main chat search function - now as a method"""
        response_parts = []

        if not all([self.passages_data, self.bi_encoder_model, self.cross_encoder_model, 
                   self.corpus_embeddings_tensor is not None]):
            response_parts.append("Error: Modelos o datos no inicializados. Revisa la consola.")
            return "\n".join(response_parts), html_path

        user_query = message.strip()
        if not user_query:
            response_parts.append("Introduce una consulta.")
            return "\n".join(response_parts), html_path

        # Stage 1: Query Analysis
        analyzed_query_dict = self.analyze_query_with_llm(user_query)
        user_query_mejorado = analyzed_query_dict.get("consulta_refinada", user_query)
        if not user_query_mejorado.strip(): 
            user_query_mejorado = user_query
        
        keywords_display_list = sorted(list(set(
            k for cat in ["quien","cuando","donde","que"] 
            for k in analyzed_query_dict.get(cat, []) 
            if isinstance(k, str) and k.strip()
        )))
        keywords_display_str = ", ".join(keywords_display_list) if keywords_display_list else "ninguna palabra clave específica"
        
        response_parts.append(f"Buscando: {{{user_query_mejorado}, [{keywords_display_str}]}}\n")
        response_parts.append("")

        # Stage 2: Retrieval
        search_terms = self.manage_search_terms(analyzed_query_dict)
        if not search_terms: 
            search_terms = [user_query_mejorado]

        all_candidate_chunks_scored = []

        for term in search_terms[:self.MAX_SEARCH_TERMS]:
            q_embedding = self.bi_encoder_model.encode(term, convert_to_tensor=True)
            
            if q_embedding.device != self.corpus_embeddings_tensor.device:
                corpus_tensor_device_equiv = self.corpus_embeddings_tensor.to(q_embedding.device)
            else:
                corpus_tensor_device_equiv = self.corpus_embeddings_tensor

            bi_hits = util.semantic_search(q_embedding, corpus_tensor_device_equiv, top_k=self.BI_ENCODER_TOP_K)
            if not bi_hits or not bi_hits[0]: 
                continue

            cross_input = [[term, self.passages_data[hit['corpus_id']]['text']] for hit in bi_hits[0]]
            cross_scores = self.cross_encoder_model.predict(cross_input, show_progress_bar=False)

            for i, hit in enumerate(bi_hits[0]):
                if cross_scores[i] > self.CROSS_ENCODER_THRESHOLD:
                    passage_info = self.passages_data[hit['corpus_id']]
                    all_candidate_chunks_scored.append({
                        'passage_data': passage_info,
                        'cross_score': float(cross_scores[i]),
                        'corpus_id': hit['corpus_id']
                    })
        
        all_candidate_chunks_scored.sort(key=lambda x: x['cross_score'], reverse=True)

        # Stage 3: Context Selection for LLM
        full_document_for_llm_data = None
        additional_chunks_for_llm_data = []
        conceptual_doc_scores = {} 
        
        for scored_chunk in all_candidate_chunks_scored:
            passage_details = scored_chunk['passage_data']
            conceptual_id = passage_details['doc_id']
            current_score = scored_chunk['cross_score']
            if conceptual_id not in conceptual_doc_scores or current_score > conceptual_doc_scores[conceptual_id][0]:
                conceptual_doc_scores[conceptual_id] = (
                    current_score, 
                    passage_details['base_document_id'], 
                    passage_details['processing_version'],
                    scored_chunk['corpus_id']
                )
        
        sorted_conceptual_docs = sorted(conceptual_doc_scores.items(), key=lambda item: item[1][0], reverse=True)

        top_conceptual_id_for_full_doc = None
        if sorted_conceptual_docs:
            top_conceptual_id_for_full_doc, (score, fname, p_version, _) = sorted_conceptual_docs[0]
            print(f"Top conceptual: '{top_conceptual_id_for_full_doc}' (File: {fname}, V: {p_version}), Score: {score}")
            full_text = self.load_full_document_by_details(fname, p_version)
            if full_text:
                full_document_for_llm_data = {'filename': fname, 'processing_version': p_version, 'text': full_text}
                print(f"Loaded full text: {fname} (V: {p_version})")
            else: 
                print(f"Failed to load full text: {fname} (V: {p_version})")
                
        # Stage 4: Generate LLM Answer
        llm_answer = self.generate_answer_with_llm(
            user_query_mejorado,
            full_document_for_llm_data
        )
        response_parts.append(f"Respuesta: {llm_answer}\n")

        # Generate triples to graph
        html_path_for_graph = self.answer_to_graph(llm_answer)

        # Stage 5: Full Transparency Output
        response_parts.append("--- CONTEXTO UTILIZADO PARA GENERAR LA RESPUESTA ---")
        
        if full_document_for_llm_data:
            response_parts.append("\nDOCUMENTO COMPLETO PRINCIPAL:")
            response_parts.append(f"  Archivo: {full_document_for_llm_data['filename']}")
            response_parts.append(f"  Versión de Procesamiento: {full_document_for_llm_data['processing_version']}")        
            response_parts.append(f"  Contenido:\n{full_document_for_llm_data['text']}")
        else:
            response_parts.append("\nNo se utilizó un documento completo principal. Reescribir la pregunta")
        
        if not full_document_for_llm_data and not additional_chunks_for_llm_data:
            response_parts.append("\n(No se proporcionó contexto específico al LLM para esta respuesta, o la recuperación falló).")

        return "\n".join(response_parts), html_path_for_graph

    def answer_to_graph(self, llm_answer):
        """Add answer information to knowledge graph"""
        print(f"Initial KG: {len(self.knowledge_graph_instance.nodes)} nodes, {len(self.knowledge_graph_instance.edges)} edges")

        # Extract triples
        extract_triples_from_text(
            text_content=llm_answer,
            current_graph=self.knowledge_graph_instance,
            client=self.llm_client_instance,
            model=self.LLM_MODEL_NAME,
            base_doc_dir_for_saving=self.GRAPH_DOCUMENT_DIRECTORY
        )

        # Save in HTML
        html_file_path = visualize_knowledge_graph(
            graph=self.knowledge_graph_instance,
            output_format="html",
            filename_prefix=self.GRAPH_FILENAME,
            output_directory=self.GRAPH_DOCUMENT_DIRECTORY, 
            show_in_browser_html=False,
            default_node_color="#A0A0A0",
            default_edge_color="#C0C0C0", 
        )

        if html_file_path:
            print(f"Generated HTML at: {html_file_path}")
            self.html_file_path = html_file_path
        
        return html_file_path

    def is_initialized(self):
        """Check if the system is properly initialized"""
        return all([
            self.passages_data, 
            self.bi_encoder_model, 
            self.cross_encoder_model, 
            self.corpus_embeddings_tensor is not None
        ])

# main.py
def main():
    print("Starting RAG program initialization...")
    
    # Create RAG system instance
    rag_system = RAGSystem()
    
    # Initialize the system
    try:
        rag_system.initialize_models_and_data()
    except Exception as e:
        print(f"Critical error during initialization: {e}")
        return

    if not rag_system.is_initialized():
        print("Critical error during initialization. Exiting.")
        return
        
    # Launch Gradio interface
    current_kg_html_path = rag_system.html_file_path
    initial_html_content = "<p>El grafo de conocimiento aparecerá aquí después de la primera consulta.</p>"

    if current_kg_html_path and os.path.exists(current_kg_html_path):
        try:
            with open(current_kg_html_path, "r", encoding="utf-8") as f:
                initial_html_content = f.read()
            print(f"Initial graph HTML loaded from: {current_kg_html_path}")
        except Exception as e:
            print(f"Error loading initial HTML from {current_kg_html_path}: {e}")
    else:
        print(f"No se encontró el archivo HTML del grafo inicial en '{current_kg_html_path}'. Se mostrará un mensaje predeterminado.")


    print("Models and data loaded. Launching Gradio interface...")
    
    with gr.Blocks() as demo:
        gr.Markdown("# Chatbot Histórico \nPregunta sobre documentos históricos. Las preguntas deben limitarse a consultas sobre personas o eventos especificos. Se mostrará todo el contexto usado para la respuesta.")
        # State to store the chat history. For 'tuples' format: List[List[str, str]]
        chat_history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=2):
                chatbot_ui = gr.Chatbot(label="Chat", bubble_full_width=False, height=600)
                input_text = gr.Textbox(placeholder="Escribe tu pregunta aquí...", show_label=False)

            with gr.Column(scale=1):
                # The HTML component for displaying the knowledge graph
                html_graph_output = gr.HTML(value=initial_html_content, label="Grafo de Conocimiento Dinámico")

        def handle_chat_interaction(user_message, current_chat_history):
            user_message = user_message.strip()
            if not user_message:
                # Handle empty input gracefully
                # No change to chat history, provide a message for HTML output, clear input
                # To keep the current graph visible, we'd need its content.
                # For simplicity, just show a message.
                # The return signature must match the `outputs` list.
                return current_chat_history, initial_html_content, current_chat_history, ""

            # 1. Get bot response and new HTML path from RAG system
            bot_response_string, new_html_file_path = rag_system.chat_search(user_message, current_chat_history)

            # 2. Update chat history
            updated_history = current_chat_history + [[user_message, bot_response_string]]

            # 3. Load HTML content for display
            html_content_for_display = "<p>El grafo de conocimiento no se actualizó para esta consulta.</p>"
            if new_html_file_path and os.path.exists(new_html_file_path):
                try:
                    with open(new_html_file_path, "r", encoding="utf-8") as f:
                        html_content_for_display = f.read()
                except Exception as e:
                    print(f"Error al leer el archivo HTML del grafo {new_html_file_path}: {e}")
                    html_content_for_display = f"<p>Error al cargar el grafo desde {new_html_file_path}: {e}</p>"
            elif new_html_file_path: # Path provided, but file does not exist
                html_content_for_display = f"<p>No se encontró el archivo HTML del grafo en: {new_html_file_path}. Revisa los logs del sistema RAG.</p>"
            else: # No path provided by rag_system.chat_search
                 html_content_for_display = f"<p>No se generó una nueva ruta para el grafo. Mostrando estado anterior o mensaje.</p>"


            # 4. Return:
            #    - updated_history (for chatbot_ui component)
            #    - html_content_for_display (for html_graph_output component)
            #    - updated_history (to update chat_history_state)
            #    - "" (to clear input_text component)
            return updated_history, html_content_for_display, updated_history, ""

        # Wire up the input submission
        input_text.submit(
            fn=handle_chat_interaction,
            inputs=[input_text, chat_history_state],
            outputs=[chatbot_ui, html_graph_output, chat_history_state, input_text]
        )

    demo.launch()

if __name__ == "__main__":
    main()