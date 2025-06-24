import os
import re
from openai import OpenAI
from llm_utils import * 
from pre_processing import *
from graph_logic import update_graph, visualize_knowledge_graph, load_knowledge_graph
from global_graph import update_global_graph
import torch 
import json
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
import pandas as pd
import gradio as gr
import networkx as nx
import datetime, uuid
from graph_search import search_graph
import csv


os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

PORT=8000
LLM_HOSTING = f"http://localhost:{PORT}/v1"
BASE_DIRECTORY = "/data/users/pfont/"
DOCUMENTS_PATH = "/data/users/pfont/processed_final"
HTML_HEIGHT=800
LOG_BASE_DIRECTORY = "/data/users/pfont/final_logs"

class RAGSystem:
    """
    A class to encapsulate all RAG system state and functionality.
    This eliminates the need for global variables.
    """
    
    def __init__(self, base_document_directory=BASE_DIRECTORY, 
                 graph_document_directory=None, conv_directory="conversations/"):
        # Configuration
        self.BASE_DOCUMENT_DIRECTORY = base_document_directory
        self.GRAPH_DOCUMENT_DIRECTORY = os.path.realpath(graph_document_directory or os.path.join(LOG_BASE_DIRECTORY, "graph/"))
        self.CONV_DIRECTORY = os.path.join(LOG_BASE_DIRECTORY, conv_directory)
        self.GRAPH_DIRECTORY_ID = os.path.join(self.GRAPH_DOCUMENT_DIRECTORY, "graph_ids/")
        self.SAVED_EMBEDDINGS_FILENAME = "corpus_embeddings_3.pt" # Saved in /data
        self.SAVED_PASSAGES_DATA = "passages_data.json"  # Saved locally
        self.LLM_MODEL_NAME = "microsoft/phi-4"
        self.GRAPH_FILENAME_BASE = "online_graph"
        self.BI_ENCODER_NAME="msmarco-bert-base-dot-v5"
        self.CROSS_ENCODER_NAME='cross-encoder/ms-marco-MiniLM-L6-v2'

        # Session Information, global graph
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        self.SESSION_GLOBAL_GRAPH_FILENAME_BASE = f"global_graph_session_{self.session_id}"
        self.session_global_graph_pkl_path = os.path.join(self.GRAPH_DOCUMENT_DIRECTORY, f"{self.SESSION_GLOBAL_GRAPH_FILENAME_BASE}.pkl")
        self.session_global_graph_html_path = os.path.join(self.GRAPH_DOCUMENT_DIRECTORY, f"{self.SESSION_GLOBAL_GRAPH_FILENAME_BASE}.html")

        # Conversation logging
        self.CONVERSATIONS_LOG_FILENAME = f"conversations_log_{self.session_id}.csv"
        self.conversations_log_path = os.path.join(self.CONV_DIRECTORY, self.CONVERSATIONS_LOG_FILENAME)
        
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
        self.knowledge_graph_global = nx.MultiDiGraph() 
        
        # Search configuration
        self.MAX_SEARCH_TERMS = 5
        self.BI_ENCODER_TOP_K = 20
        self.CROSS_ENCODER_THRESHOLD = 0.0

        # Answer Flags
        self.RAG_ANSWER = False
        self.GRAPH_ANSWER = False
        self.return_info = {}

        # Initialize conversation log file
        self._initialize_conversation_log()

    def initialize_models_and_data(self, 
                                 bi_encoder_name=None, 
                                 cross_encoder_name=None, 
                                 passages_filename=None):
        """Initialize all models and data"""
        bi_encoder_name = self.BI_ENCODER_NAME if not bi_encoder_name else bi_encoder_name
        cross_encoder_name = self.CROSS_ENCODER_NAME if not cross_encoder_name else cross_encoder_name
        passages_filename = self.SAVED_PASSAGES_DATA if not passages_filename else passages_filename
        
        # DEFAULT CONVERSATION GRAPH (can be overridden by conversation_id later)
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
        
        # TEXT CHUNKS (passage data)
        self._load_or_create_passages(passages_filename)
        
        # EMBEDDING MODELS
        self._initialize_embedding_models(bi_encoder_name, cross_encoder_name)
        
        # CORPUS EMBEDDINGS (embeddings)
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

    # Returns the text for the processed version of that id
    def load_full_document_by_details(self, document_id):
        filename = f"rsc37_rsc176_{document_id}.txt"
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

    def analyze_query_with_llm(self, user_query, history=None):
        # TODO:  Allow to go back deeper (not only last)
        # Give only history in case RAG_ANSWER was True
        add_history = False
        if history and len(history) > 1:
            question = history[-2][0]
            answer = history[-2][1]
            if question and answer: 
                answer = answer.split('\n\n**Respuesta:**\n')[0].strip() 
                answer = answer.replace("\n", " ")
                formatted_history = f"{answer}"
                add_history = True
                history_prompt = f"""Solo en caso EXTREMADAMENTE necesario, puedes usar el historial de antiguas entidades clave: {formatted_history}\n"""
                print(f"Adding to prompt {history_prompt}")

        """Analyze query with LLM to extract entities"""
        prompt = f"""
        Analiza la siguiente consulta en español e identifica las entidades clave.
        Ten en cuenta que se usaran para Retrieval, no añadas conceptos generales. 

        Consulta del usuario: "{user_query}"

        Extrae y clasifica las siguientes entidades:
        1. QUIÉN (Personas): Nombres de personas, grupos, organizaciones, etc. LA MAS IMPORTANTE
        2. CUÁNDO (Tiempo): Fechas, períodos, siglos, años, meses, días, etc.
        3. DÓNDE (Lugar): Ubicaciones geográficas, países, ciudades, regiones, etc.
        4. QUÉ (Tema): El tema principal, eventos, conceptos, términos específicos, etc.
        """
        if add_history:
            prompt += history_prompt
        
        prompt += """
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
        
        print(f"USING PROMPT {prompt}")
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

    def generate_answer_from_graph(self, user_query, graph_triplets: list[str]):
        default_no_info_response = (False, "No se proporcionó contexto del grafo para procesar la pregunta.")
        default_llm_error_response = (False, "Error al procesar la respuesta del LLM o formato JSON inválido.")
        
        if not graph_triplets or not isinstance(graph_triplets, list):
            return default_no_info_response

        context_str = "\n".join(graph_triplets)

        prompt = f"""
        Eres un asistente de IA experto en razonamiento sobre grafos de conocimiento extraídos de documentos históricos en español.
        Tu tarea es responder a la pregunta del usuario basándote *únicamente* en los hechos proporcionados en el "CONTEXTO DEL GRAFO".
        El contexto se presenta como una lista de tripletas con el formato: Sujeto --[Relación]--> Objeto.

        Debes responder directamente. No infieras ni inventes información que no esté explícitamente en las tripletas.
        Si la información no se puede deducir directamente de las tripletas, debes indicarlo.

        Tu respuesta DEBE ser un objeto JSON válido. ASEGÚRATE de que la salida sea *únicamente* el JSON y nada más.
        El JSON debe tener la siguiente estructura:
        {{
            "informacion_encontrada": <boolean>, // true si la respuesta se basa en los hechos del grafo, false si el contexto no es relevante o no contiene la respuesta.
            "respuesta": "<string>" // La respuesta a la pregunta. Si informacion_encontrada es false, este campo debe indicar que la información no fue encontrada en los hechos proporcionados.
        }}

        Pregunta del Usuario: "{user_query}"

        CONTEXTO DEL GRAFO:
        ---
        {context_str}
        ---

        Respuesta JSON:
        """

        # 3. LLM call and JSON parsing (this logic remains the same as it's robust)
        if self.llm_client_instance:
            try:
                llm_response_str = query_llm(self.llm_client_instance, self.LLM_MODEL_NAME, prompt)
                
                # Robust JSON extraction
                json_start = llm_response_str.find('{')
                json_end = llm_response_str.rfind('}')
                
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str_cleaned = llm_response_str[json_start : json_end+1]
                    parsed_json = json.loads(json_str_cleaned)
                    
                    # Validate JSON structure and types
                    if "informacion_encontrada" in parsed_json and "respuesta" in parsed_json and \
                       isinstance(parsed_json["informacion_encontrada"], bool) and \
                       isinstance(parsed_json["respuesta"], str):
                        return parsed_json["informacion_encontrada"], parsed_json["respuesta"]
                    else:
                        print(f"Error: El JSON devuelto no tiene la estructura o tipos esperados. Respuesta LLM: {llm_response_str}")
                        return default_llm_error_response
                else:
                    print(f"Error: No se pudo extraer un JSON válido de la respuesta del LLM. Respuesta LLM: {llm_response_str}")
                    return default_llm_error_response
                    
            except json.JSONDecodeError as e:
                print(f"Error al decodificar JSON de la respuesta del LLM: {e}. Respuesta LLM: {llm_response_str}")
                return default_llm_error_response
            except Exception as e:
                print(f"Error inesperado al interactuar con el LLM: {e}")
                return default_llm_error_response
        else:
            # This case handles if self.llm_client_instance is not set
            return default_no_info_response

    def generate_answer_with_llm(self, user_query, full_doc_context):
        """Generate answer using LLM with provided context"""
        default_no_info_response = (False, "No se proporcionó contexto para procesar la pregunta.")
        default_llm_error_response = (False, "Error al procesar la respuesta del LLM o formato JSON inválido.")
         
        contexts_for_prompt = []
        if full_doc_context and full_doc_context.get('text'):
            contexts_for_prompt.append(
                f"--- DOCUMENTO COMPLETO PRINCIPAL (Archivo: {full_doc_context.get('filename', 'Desconocido')}, Versión de procesamiento: {full_doc_context.get('processing_version', 'N/A')}) ---\n"
                f"{full_doc_context['text']}\n"
                f"--- FIN DEL DOCUMENTO COMPLETO PRINCIPAL ---"
            )
        
        if not contexts_for_prompt:
            return default_no_info_response

        context_str = "\n\n".join(contexts_for_prompt)
        
        prompt = f"""
        Eres un asistente de IA experto en documentos históricos en español.
        Tu tarea es responder a la pregunta del usuario basándote *únicamente* en el contexto del "DOCUMENTO COMPLETO PRINCIPAL".
        Debes ser conciso, responder directamente y sintetizar la información. No inventes información que no esté explícitamente en el texto.
        Puedes asumir pequeñas discrepancias tipográficas (por ejemplo, errores comunes de OCR en nombres propios), pero no debes inferir ni inventar nada fuera del texto.
        No menciones "DOCUMENTO COMPLETO PRINCIPAL" en tu respuesta final.

        Tu respuesta DEBE ser un objeto JSON válido. ASEGÚRATE de que la salida sea *únicamente* el JSON y nada más.
        El JSON debe tener la siguiente estructura:
        {{
            "informacion_encontrada": <boolean>, // true si la respuesta se basa en el contexto y la información fue hallada, false en caso contrario.
            "respuesta": "<string>" // La respuesta a la pregunta. Si informacion_encontrada es false, este campo debe indicar que la información no fue encontrada o el contexto no es relevante.
        }}

        Pregunta del Usuario: "{user_query}"

        Contextos Proporcionados:
        {context_str}

        Respuesta JSON:
        """
        if self.llm_client_instance:
            try:
                llm_response_str = query_llm(self.llm_client_instance, self.LLM_MODEL_NAME, prompt)
                json_start = llm_response_str.find('{')
                json_end = llm_response_str.rfind('}')
                
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str_cleaned = llm_response_str[json_start : json_end+1]
                    parsed_json = json.loads(json_str_cleaned)
                    if "informacion_encontrada" in parsed_json and "respuesta" in parsed_json and \
                       isinstance(parsed_json["informacion_encontrada"], bool) and \
                       isinstance(parsed_json["respuesta"], str):
                        return parsed_json["informacion_encontrada"], parsed_json["respuesta"]
                    else:
                        print(f"Error: El JSON devuelto no tiene la estructura o tipos esperados. Respuesta LLM: {llm_response_str}")
                        return default_llm_error_response
                else:
                    print(f"Error: No se pudo extraer un JSON válido de la respuesta del LLM. Respuesta LLM: {llm_response_str}")
                    return default_llm_error_response 
                    
            except json.JSONDecodeError as e:
                print(f"Error al decodificar JSON de la respuesta del LLM: {e}. Respuesta LLM: {llm_response_str}")
                return default_llm_error_response 
            except Exception as e:
                print(f"Error inesperado al interactuar con el LLM: {e}")
                return default_llm_error_response 
        else:
            return default_no_info_response 

    def manage_search_terms(self, analyzed_query_dict):
        """Extract and manage search terms from analyzed query"""
        search_terms = []

        # Add "quien" terms first
        quien_terms = analyzed_query_dict.get("quien", [])
        if isinstance(quien_terms, list):
            search_terms.extend(k for k in quien_terms if isinstance(k, str) and k.strip())
        elif isinstance(quien_terms, str) and quien_terms.strip():
            search_terms.append(quien_terms.strip())

        # Add "consulta_refinada" next
        consulta_refinada = analyzed_query_dict.get("consulta_refinada", "")
        if isinstance(consulta_refinada, str) and consulta_refinada.strip():
            search_terms.append(consulta_refinada.strip())

        # If no "quien" terms, add other categories
        if not quien_terms:
            for key_type in ["cuando", "donde", "que"]:
                keywords = analyzed_query_dict.get(key_type, [])
                if isinstance(keywords, list):
                    search_terms.extend(k for k in keywords if isinstance(k, str) and k.strip())
                elif isinstance(keywords, str) and keywords.strip():
                    search_terms.append(keywords.strip())
        return search_terms

    def chat_search(self, message, history, conversation_id=None):
        response_parts = []
        self.return_info = {}
        self.RAG_ANSWER = False
        self.GRAPH_ANSWER = False

        graph_file_suffix = f"_{conversation_id}" if conversation_id else ""
        expected_html_path_for_graph = os.path.join(self.GRAPH_DIRECTORY_ID, f"{self.GRAPH_FILENAME_BASE}{graph_file_suffix}.html")
        if os.path.exists(expected_html_path_for_graph):
            html_path_for_graph = expected_html_path_for_graph
        else:
            html_path_for_graph = None

        # Make sure everything is loaded
        if not self.is_initialized():
            response_parts.append("Error: Modelos o datos no inicializados. Revisa la consola.")
            self.return_info["llm_answer"] = response_parts
            return self.return_info, html_path_for_graph # Return current/last known path

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
            self.return_info["llm_answer"] = response_parts
            return self.return_info, html_path_for_graph # Return current/last known path

        # Stage 1: Query Analysis
        analyzed_query_dict = self.analyze_query_with_llm(user_query, history)
        user_query_mejorado = analyzed_query_dict.get("consulta_refinada", user_query)
        if not user_query_mejorado.strip(): 
            user_query_mejorado = user_query
        
        keywords_display_list = sorted(list(set(
            k for cat in ["quien","cuando","donde","que"] 
            for k in analyzed_query_dict.get(cat, []) 
            if isinstance(k, str) and k.strip()
        )))
        keywords_display_str = ", ".join(keywords_display_list) if keywords_display_list else "ninguna palabra clave específica"
        
        response_parts.append(f"\n\n**Buscando:** \{{{user_query_mejorado}, [{keywords_display_str}]}}\n")
        response_parts.append("")

        # Stage 2: Retrieval
        search_terms_essential = self.manage_search_terms(analyzed_query_dict)
        if not search_terms_essential: 
            search_terms_essential = [user_query_mejorado]
        
        all_candidate_chunks_scored = []

        for term in search_terms_essential[:self.MAX_SEARCH_TERMS]:
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
            full_text = self.load_full_document_by_details(str(top_conceptual_id_for_full_doc))
            if full_text:
                full_document_for_llm_data = {'filename': fname, 'processing_version': p_version, 'text': full_text}
                self.return_info = full_document_for_llm_data
                self.return_info["id"] = top_conceptual_id_for_full_doc
                print(f"Loaded full text from document: {top_conceptual_id_for_full_doc}")
            else: 
                print(f"Failed to load full text: {fname} (V: {p_version})")
        
                
        # Stage 4: Generate LLM Answer
        (self.RAG_ANSWER, llm_answer) = self.generate_answer_with_llm(
            user_query_mejorado,
            full_document_for_llm_data
        )
        # Is there context? 
        if not self.RAG_ANSWER:
            print("LLM identified no relevant context")

        response_parts.append(f"\n\n**Respuesta:**\n {llm_answer}\n")

        # NEW STAGE: GRAPH SEARCH
        if keywords_display_list and self.knowledge_graph_global and self.knowledge_graph_global.number_of_nodes() > 0:
            context, graph_search_success = search_graph(self.knowledge_graph_global, keywords_display_list)
            if graph_search_success:
                print(f"Graph search successful. Context found: {len(context)} snippets.")

                graph_answer_found_flag, graph_llm_answer_text = self.generate_answer_from_graph(
                    user_query_mejorado, 
                    context
                )

                if graph_answer_found_flag:
                    print(f"Graph-based LLM answer generated: {graph_llm_answer_text}")
                    self.GRAPH_ANSWER = True # Set the flag
                    self.return_info["graph_answer"] = graph_llm_answer_text
                    self.return_info["graph_context"] = context # Store the list of contexts
            else:
                print("Graph-based LLM found no relevant information in the provided graph context.")

        # Generate triples to graph
        if self.RAG_ANSWER: # Only add triplets if there is an answer
            # NORMAL:
            newly_generated_html_path, triplets_added = self.answer_to_graph(llm_answer, conversation_id=conversation_id) # PASAR conversation_id
            if newly_generated_html_path and os.path.exists(newly_generated_html_path):
                 html_path_for_graph = newly_generated_html_path

            # ADD TO GLOBAL:
            document_id_for_global_graph = self.return_info.get("id")
            if triplets_added and document_id_for_global_graph:
                self.update_global_knowledge_graph(triplets_added, str(document_id_for_global_graph))
            elif not triplets_added:
                print("No processed triplets from answer_to_graph to add to global graph.")
            elif not document_id_for_global_graph:
                print("No document_id available for global graph update.")

        # Stage 5: Full Transparency Output
        if not full_document_for_llm_data: 
            response_parts.append("\nNo se identificó un documento principal como contexto. Reescribir la pregunta")

        self.return_info["llm_answer"] = response_parts
        
        # Log conversation details to CSV
        self._log_conversation(
            conversation_id=conversation_id,
            question=user_query,
            answer_parts=response_parts,
            analyzed_query_dict=analyzed_query_dict,
            search_terms=search_terms_essential,
            refined_query=user_query_mejorado,
            document_info=full_document_for_llm_data,
            cross_score=sorted_conceptual_docs[0][1][0] if sorted_conceptual_docs else None
        )

        return self.return_info, html_path_for_graph

    def update_global_knowledge_graph(self, processed_triplets, document_id):
        """Updates the global knowledge graph with the given triplets and document ID."""
        if not processed_triplets:
            print("Global Graph Update: No triplets provided.")
            return
        if not document_id:
            print("Global Graph Update: No document_id provided.")
            return

        print(f"Updating global graph with {len(processed_triplets)} triplets. Document ID: {document_id}")

        self.knowledge_graph_global = update_global_graph(
            processed_triplets=processed_triplets,
            global_graph=self.knowledge_graph_global,
            document_id=str(document_id),
            filepath=self.session_global_graph_pkl_path
        )

        visualize_knowledge_graph(
            graph=self.knowledge_graph_global,
            output_format="html",
            filename_prefix=self.SESSION_GLOBAL_GRAPH_FILENAME_BASE, # This will create "global_knowledge_graph.html"
            output_directory=self.GRAPH_DOCUMENT_DIRECTORY,
            show_in_browser_html=False, # Set to True if you want it to pop up, False for server use
            html_height=str(HTML_HEIGHT)+"px"
        )
    
    def answer_to_graph(self, llm_answer, conversation_id=None):
        """Add answer information to knowledge graph"""
        print(f"Initial KG: {len(self.knowledge_graph_instance.nodes)} nodes, {len(self.knowledge_graph_instance.edges)} edges")

        graph_file_suffix = f"_{conversation_id}" if conversation_id else ""
        current_graph_filename_prefix = f"{self.GRAPH_FILENAME_BASE}{graph_file_suffix}"

        #PKL FILEPATH:
        filepath = os.path.join(self.GRAPH_DIRECTORY_ID, f"{current_graph_filename_prefix}.pkl")

        # Extract triples, process them and save new graph
        self.knowledge_graph_instance, triplets_added = update_graph( 
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
            return new_html_file_path, triplets_added
        else:
            print(f"Failed to generate or find HTML at: {new_html_file_path}. Returning None.")
            return None, triplets_added # Devolver None si la generación falla

    def _initialize_conversation_log(self):
        """Initialize the conversation log CSV file with headers"""
        headers = [
            'session_id',
            'conversation_id', 
            'question',
            'answer',
            'rag_answer_flag',
            'graph_answer_flag',
            'graph_answer_text',
            'quien_entities',
            'cuando_entities', 
            'donde_entities',
            'que_entities',
            'search_terms',
            'refined_query',
            'document_id',
            'document_filename',
            'processing_version',
            'cross_encoder_score',
            'graph_context_snippets'
        ]
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(self.conversations_log_path):
            try:
                with open(self.conversations_log_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(headers)
                print(f"Initialized conversation log at: {self.conversations_log_path}")
            except Exception as e:
                print(f"Error initializing conversation log: {e}")

    def _log_conversation(self, conversation_id, question, answer_parts, analyzed_query_dict, 
                         search_terms, refined_query, document_info=None, cross_score=None):
        """Log conversation details to CSV file"""        
        try:
            # Extract answer text from response_parts
            answer_text = ""
            if isinstance(answer_parts, list):
                # Join all parts and clean up the response formatting
                full_response = "".join(answer_parts)
                # Extract just the answer part, removing search info
                if "**Respuesta:**" in full_response:
                    answer_text = full_response.split("**Respuesta:**")[1].strip()
                    if answer_text.startswith("\n"):
                        answer_text = answer_text[1:].strip()
                else:
                    answer_text = full_response.strip()
            else:
                answer_text = str(answer_parts)

            # Prepare entity lists
            quien_entities = ", ".join(analyzed_query_dict.get("quien", []))
            cuando_entities = ", ".join(analyzed_query_dict.get("cuando", []))
            donde_entities = ", ".join(analyzed_query_dict.get("donde", []))
            que_entities = ", ".join(analyzed_query_dict.get("que", []))
            search_terms_str = ", ".join(search_terms) if search_terms else ""
            
            # Graph answer info
            graph_answer_text = self.return_info.get("graph_answer", "")
            graph_context_list = self.return_info.get("graph_context", "")
            graph_context_str= " | ".join(graph_context_list)
            graph_context_str = graph_context_str.replace("---", " | ")
            graph_context_str = graph_context_str.replace("\n", " | ")
            
            # Document info
            doc_id = document_info.get("id", "") if document_info else ""
            doc_filename = document_info.get("filename", "") if document_info else ""
            proc_version = document_info.get("processing_version", "") if document_info else ""
            
            row_data = [
                self.session_id,
                conversation_id or "",
                question,
                answer_text,
                self.RAG_ANSWER,
                self.GRAPH_ANSWER,
                graph_answer_text,
                quien_entities,
                cuando_entities,
                donde_entities,
                que_entities,
                search_terms_str,
                refined_query,
                doc_id,
                doc_filename,
                proc_version,
                cross_score if cross_score else "",
                graph_context_str
            ]
            
            # Append to CSV file
            with open(self.conversations_log_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
                
            print(f"Logged conversation to: {self.conversations_log_path}")
            
        except Exception as e:
            print(f"Error logging conversation: {e}")

    def is_initialized(self):
        """Check if the system is properly initialized"""
        return all([
            self.passages_data, 
            self.bi_encoder_model, 
            self.cross_encoder_model, 
            self.corpus_embeddings_tensor is not None
        ])

