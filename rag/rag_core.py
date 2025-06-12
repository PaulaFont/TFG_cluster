import os
import re
from openai import OpenAI
from llm_utils import * 
from pre_processing import *
from graph_logic import update_graph, visualize_knowledge_graph, load_knowledge_graph
import torch 
import json
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
import pandas as pd
import gradio as gr
import networkx as nx

os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

PORT=8000
LLM_HOSTING = f"http://localhost:{PORT}/v1"
BASE_DIRECTORY = "/data/users/pfont/"
DOCUMENTS_PATH = "/data/users/pfont/final_documents"
HTML_HEIGHT=800


class RAGSystem:
    """
    A class to encapsulate all RAG system state and functionality.
    This eliminates the need for global variables.
    """
    
    def __init__(self, base_document_directory=BASE_DIRECTORY, 
                 graph_document_directory=None):
        # Configuration
        self.BASE_DOCUMENT_DIRECTORY = base_document_directory
        self.GRAPH_DOCUMENT_DIRECTORY = os.path.realpath(graph_document_directory or os.path.join(base_document_directory, "graph/"))
        self.GRAPH_DIRECTORY_ID = os.path.join(self.GRAPH_DOCUMENT_DIRECTORY, "graph_ids/")
        self.SAVED_EMBEDDINGS_FILENAME = "corpus_embeddings_2.pt" # Saved in /data
        self.SAVED_PASSAGES_DATA = "passages_data.json"  # Saved locally
        self.LLM_MODEL_NAME = "microsoft/phi-4"
        self.GRAPH_FILENAME_BASE = "online_graph"
        self.BI_ENCODER_NAME="msmarco-bert-base-dot-v5"
        self.CROSS_ENCODER_NAME='cross-encoder/ms-marco-MiniLM-L6-v2'
        
        # Create directories
        os.makedirs(self.GRAPH_DOCUMENT_DIRECTORY, exist_ok=True)
        os.makedirs(self.GRAPH_DIRECTORY_ID, exist_ok=True)
        if not os.path.exists(self.BASE_DOCUMENT_DIRECTORY):
            print(f"WARNING: BASE_DOCUMENT_DIRECTORY '{self.BASE_DOCUMENT_DIRECTORY}' does not exist.")
        
        # State variables
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

        # Answer Flags
        self.RAG_ANSWER = False
        self.GRAPH_ANSWER = False

    def initialize_models_and_data(self, 
                                 bi_encoder_name=None, 
                                 cross_encoder_name=None, 
                                 passages_filename=None):
        """Initialize all models and data"""
        bi_encoder_name = self.BI_ENCODER_NAME if not bi_encoder_name else bi_encoder_name
        cross_encoder_name = self.CROSS_ENCODER_NAME if not cross_encoder_name else cross_encoder_name
        passages_filename = self.SAVED_PASSAGES_DATA if not passages_filename else passages_filename
        
        # GRAPH
        kg_filepath = os.path.join(self.GRAPH_DOCUMENT_DIRECTORY, self.GRAPH_FILENAME_BASE + ".pkl")
        self.knowledge_graph_instance = load_knowledge_graph(kg_filepath)

        # LLM
        if not start_llm_server(self.LLM_MODEL_NAME, port=PORT):
            self.llm_client_instance = None
            print("LLM Server failed to start. LLM functionalities will be unavailable.")
        else:
            self.llm_client_instance = OpenAI(
                base_url=LLM_HOSTING,
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
        #TODO: Manage versions. At the moment just returns the one with more score
        """Load the full text content of a document"""
        if not filename or not processing_version:
            print("Error: Filename or processing_version missing.")
            return None

        filepath = os.path.join(DOCUMENTS_PATH, filename)

        if not os.path.exists(filepath):
            print(f"Error: Document file '{filepath}' not found.")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading document file '{filepath}': {e}")
            return None

    def analyze_query_with_llm(self, user_query):
        """Analyze query with LLM to extract entities"""
        #TODO: I changed the prompt asking it to be more specific. Check
        prompt = f"""
        Analiza la siguiente consulta en español e identifica las entidades clave.
        Ten en cuenta que se usaran para Retrieval, no añadas conceptos generales. 

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
        # Make sure it returns "0" when there is no context. 
        contexts_for_prompt = []
        if full_doc_context and full_doc_context.get('text'):
            contexts_for_prompt.append(
                f"--- DOCUMENTO COMPLETO PRINCIPAL (Archivo: {full_doc_context.get('filename', 'Desconocido')}, Versión de procesamiento: {full_doc_context.get('processing_version', 'N/A')}) ---\n"
                f"{full_doc_context['text']}\n"
                f"--- FIN DEL DOCUMENTO COMPLETO PRINCIPAL ---"
            )
        
        if not contexts_for_prompt:
            return "0" 

        context_str = "\n\n".join(contexts_for_prompt)
        
        prompt = f"""
        Eres un asistente de IA experto en documentos históricos en español.
        Responde a la pregunta del usuario basándote *únicamente* en el contexto del "DOCUMENTO COMPLETO PRINCIPAL".
        Sé conciso y responde directamente. Sintetiza la información. No inventes nada.
        Si la información no está en los textos, indícalo contestando un "0" y nada más.  
        No menciones "DOCUMENTO COMPLETO PRINCIPAL" o "FRAGMENTOS ADICIONALES" en tu respuesta final.

        Pregunta del Usuario: "{user_query}"

        Contextos Proporcionados:
        {context_str}

        Respuesta Concisa:
        """
        if self.llm_client_instance:
            return query_llm(self.llm_client_instance, self.LLM_MODEL_NAME, prompt)
        else:
            return "0"

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

    def chat_search(self, message, history, conversation_id=None):
        html_path_for_graph = None 
        response_parts = []

        # Make sure everything is loaded
        if not self.is_initialized():
            response_parts.append("Error: Modelos o datos no inicializados. Revisa la consola.")
            return "\n".join(response_parts), html_path_for_graph # Return current/last known path

        # MANAGE CHANGING THE GRAPH INSTANCE TO THE ONE FOR THE CONVERSATION
        if conversation_id:
            conv_graph_pkl_filename = f"{self.GRAPH_FILENAME_BASE}_{conversation_id}.pkl"
            conv_graph_path = os.path.join(self.GRAPH_DIRECTORY_ID, conv_graph_pkl_filename)
            self.knowledge_graph_instance = load_knowledge_graph(conv_graph_path) # load_knowledge_graph is from graph_logic
            print(f"Loaded graph for conversation {conversation_id} from {conv_graph_path}. Nodes: {len(self.knowledge_graph_instance.nodes)}, Edges: {len(self.knowledge_graph_instance.edges)}")
        else:
            default_graph_path = os.path.join(self.GRAPH_DOCUMENT_DIRECTORY, self.GRAPH_FILENAME_BASE + ".pkl")
            self.knowledge_graph_instance = load_knowledge_graph(default_graph_path)
            print(f"No conversation_id in chat_search, loaded default graph. Nodes: {len(self.knowledge_graph_instance.nodes)}, Edges: {len(self.knowledge_graph_instance.edges)}")
        # --- END: Load conversation-specific graph ---


        user_query = message.strip()
        if not user_query:
            response_parts.append("Introduce una consulta.")
            return "\n".join(response_parts), html_path_for_graph # Return current/last known path

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
        search_terms = self.manage_search_terms(analyzed_query_dict) #TODO: add something to order them by more relevant ??
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
                    passage_details['original_filename'], 
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
        # Is there context? Returns "0" when there is no satisfactory answer. 
        if llm_answer == "0":
            llm_answer = "No se ha encontrado contexto suficientemente relevante para contestar esta pregunta."
            self.RAG_ANSWER = False
        else: 
            self.RAG_ANSWER = True

        response_parts.append(f"Respuesta: {llm_answer}\n")

        # Generate triples to graph
        if self.RAG_ANSWER: # Only add triplets if there is an answer
            newly_generated_html_path = self.answer_to_graph(llm_answer, conversation_id=conversation_id) # PASAR conversation_id
            if newly_generated_html_path and os.path.exists(newly_generated_html_path):
                 html_path_for_graph = newly_generated_html_path
            # else: html_path_for_graph remains None, Gradio mostrará "No hay grafo"

        # Stage 5: Full Transparency Output
        response_parts.append("--- CONTEXTO UTILIZADO PARA GENERAR LA RESPUESTA ---")
        
        if full_document_for_llm_data: #TODO: Make it more clear
            response_parts.append("\nDOCUMENTO COMPLETO PRINCIPAL:")
            response_parts.append(f"  Archivo: {full_document_for_llm_data['filename']}")
            response_parts.append(f"  Versión de Procesamiento: {full_document_for_llm_data['processing_version']}")        
            response_parts.append(f"  Contenido:\n{full_document_for_llm_data['text']}")
        else:
            response_parts.append("\nNo se utilizó un documento completo principal. Reescribir la pregunta")
        
        if not full_document_for_llm_data and not additional_chunks_for_llm_data:
            response_parts.append("\n(No se proporcionó contexto específico al LLM para esta respuesta, o la recuperación falló).")

        return "\n".join(response_parts), html_path_for_graph

    def answer_to_graph(self, llm_answer, conversation_id=None):
        """Add answer information to knowledge graph"""
        print(f"Initial KG: {len(self.knowledge_graph_instance.nodes)} nodes, {len(self.knowledge_graph_instance.edges)} edges")

        graph_file_suffix = f"_{conversation_id}" if conversation_id else ""
        current_graph_filename_prefix = f"{self.GRAPH_FILENAME_BASE}{graph_file_suffix}"

        #PKL FILEPATH:
        filepath = os.path.join(self.GRAPH_DIRECTORY_ID, f"{current_graph_filename_prefix}.pkl")

        # Extract triples, process them and save new graph
        self.knowledge_graph_instance = update_graph( 
            text_content=llm_answer,
            current_graph=self.knowledge_graph_instance,
            client=self.llm_client_instance,
            filepath=filepath,
            model=self.LLM_MODEL_NAME,
        )

        # Save in HTML
        new_html_file_path = visualize_knowledge_graph(
            graph=self.knowledge_graph_instance,
            output_format="html",
            filename_prefix=current_graph_filename_prefix,
            output_directory=self.GRAPH_DIRECTORY_ID, 
            show_in_browser_html=False,
            default_node_color="#A0A0A0",
            default_edge_color="#C0C0C0", 
        )

        if new_html_file_path and os.path.exists(new_html_file_path):
            print(f"Generated HTML at: {new_html_file_path}")
            # self.html_file_path ya no se actualiza aquí globalmente, se devuelve la ruta específica
            return new_html_file_path
        else:
            print(f"Failed to generate or find HTML at: {new_html_file_path}. Returning None.")
            return None # Devolver None si la generación falla

    def is_initialized(self):
        """Check if the system is properly initialized"""
        return all([
            self.passages_data, 
            self.bi_encoder_model, 
            self.cross_encoder_model, 
            self.corpus_embeddings_tensor is not None
        ])

