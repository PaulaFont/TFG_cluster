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

passages_data = []
bi_encoder_model = None
cross_encoder_model = None
corpus_embeddings_tensor = None
llm_client_instance = None
knowledge_graph_instance = None 

# KG_FILENAME = "online_knowledge_graph.pkl"
SAVED_EMBEDDINGS_FILENAME = "corpus_embeddings.pt"
BASE_DOCUMENT_DIRECTORY = "/data/users/pfont/" 
if not os.path.exists(BASE_DOCUMENT_DIRECTORY):
    print(f"WARNING: BASE_DOCUMENT_DIRECTORY '{BASE_DOCUMENT_DIRECTORY}' does not exist. Full document loading will fail.")


def initialize_models_and_data(llm_model_name="microsoft/phi-4", bi_encoder_name="msmarco-bert-base-dot-v5", cross_encoder_name='cross-encoder/ms-marco-MiniLM-L6-v2', passages_filename="passages_data.json"):
    global passages_data, bi_encoder_model, cross_encoder_model, corpus_embeddings_tensor, llm_client_instance
    global knowledge_graph_instance # Add knowledge_graph global

    kg_filepath = os.path.join(BASE_DOCUMENT_DIRECTORY, KG_FILENAME)
    knowledge_graph_instance = load_knowledge_graph(kg_filepath)

    if not start_llm_server(llm_model_name, port=8000): # Assuming this handles its own errors
        llm_client_instance = None
        print("LLM Server failed to start. LLM functionalities will be unavailable.")
    else:
        llm_client_instance = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123" 
        )
        print("LLM Client initialized.")
    
    print("Creating/Loading passages and metadata...")
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

    if not passages_data:
        print("No passages were loaded or created. Critical error.")
        return

    # Validate essential keys in passages_data
    for i, p in enumerate(passages_data):
        if 'text' not in p or not p['text'].strip():
            print(f"Warning: Passage {i} is missing 'text' or has empty text.")
        if 'base_document_id' not in p:
            print(f"Warning: Passage {i} is missing 'base_document_id' (filename).")
            p['base_document_id'] = f"unknown_filename_{i}" # Placeholder
        if 'processing_version' not in p:
            print(f"Warning: Passage {i} is missing 'processing_version'. This is needed for full document path.")
            # Attempt to infer from a path if available, or use a default
            p['processing_version'] = "unknown_version" # Placeholder
            if 'file_path' in p and "out_llm_" in p['file_path']:
                path_parts = p['file_path'].split(os.sep)
                for part in path_parts:
                    if part.startswith("out_llm_"):
                        p['processing_version'] = part.replace("out_llm_","")
                        break
        # Derive conceptual_doc_id (assumes base_document_id is the filename without versioning info in its name itself)
        # If base_document_id can contain version info, extract_conceptual_id needs to be more robust
        p['conceptual_doc_id'] = extract_conceptual_id_from_filename(p.get('base_document_id', ''))


    passage_texts = [p['text'] for p in passages_data if 'text' in p and p['text'].strip()]
    if not passage_texts:
        print("Passages data contains no valid text entries. Critical error.")
        return

    print("Loading embedding models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    try:
        bi_encoder_model = SentenceTransformer(bi_encoder_name, device=device)
        cross_encoder_model = CrossEncoder(cross_encoder_name, device=device) 
    except Exception as e:
        print(f"Error loading sentence-transformer models: {e}")
        return
    
    
    saved_embeddings_path = os.path.join(BASE_DOCUMENT_DIRECTORY, SAVED_EMBEDDINGS_FILENAME)
    # --- Attempt to load pre-computed embeddings and processed passages_data ---
    loaded_from_cache = False
    if os.path.exists(saved_embeddings_path):
        print(f"Found pre-computed embeddings at {saved_embeddings_path}")
        try:
            print("Loading corpus embeddings...")
            corpus_embeddings_tensor = torch.load(saved_embeddings_path, map_location=device) # Load to current device
            if corpus_embeddings_tensor.shape[0] == len(passage_texts):
                print(f"Loaded corpus embeddings. Shape: {corpus_embeddings_tensor.shape}, Device: {corpus_embeddings_tensor.device}")
                loaded_from_cache = True
                print("Successfully loaded embeddings from cache.")
            else:
                print(f"Mismatch! Loaded embeddings count ({corpus_embeddings_tensor.shape[0]}) "
                      f"does not match current passage texts count ({len(passage_texts)}). "
                      f"Will regenerate embeddings.")
                corpus_embeddings_tensor = None # Invalidate loaded tensor
                loaded_from_cache = False # Force regeneration

        except Exception as e:
            print(f"Error loading from cache: {e}. Will regenerate.")
            loaded_from_cache = False
            corpus_embeddings_tensor = None
    else:
        print("Pre-computed files not found. Will generate new ones.")

    if not loaded_from_cache:
        print("Creating embeddings for all passages...")
        corpus_embeddings_tensor = bi_encoder_model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=True)
        print(f"Corpus embeddings created on device: {corpus_embeddings_tensor.device}. Shape: {corpus_embeddings_tensor.shape}")
        if corpus_embeddings_tensor is not None: # Only save if generation was successful
            try:
                if not os.path.exists(BASE_DOCUMENT_DIRECTORY): # Double check dir exists before saving
                    print(f"Target save directory {BASE_DOCUMENT_DIRECTORY} does not exist. Skipping save.")
                else:
                    print(f"Saving new corpus embeddings to {saved_embeddings_path}...")
                    torch.save(corpus_embeddings_tensor, saved_embeddings_path)
                    print("Embeddings saved successfully.")
            except Exception as e:
                print(f"Error saving new corpus embeddings to {saved_embeddings_path}: {e}")


    if corpus_embeddings_tensor is None: # Final check
        print("Critical error: Corpus embeddings are not available after initialization attempt.")
        return
    print("Initialization complete.")


def extract_conceptual_id_from_filename(filename):
    """
    Extracts a conceptual document ID from a filename.
    If your filenames already are the conceptual ID (e.g., "document_XYZ.txt" and version is only in folder),
    then this can be simpler.
    This example assumes filenames might also have some versioning/ocr tags to strip.
    """
    if not filename:
        return "unknown_conceptual_doc"

    return os.path.splitext(filename)[0] # Fallback


def load_full_document_by_details(filename, processing_version):
    """
    Loads the full text content of a document given its filename and processing_version.
    Constructs path: BASE_DOCUMENT_DIRECTORY / "out_llm_{processing_version}" / filename
    """
    if not filename or not processing_version:
        print("Error: Filename or processing_version missing for load_full_document_by_details.")
        return None
    if not BASE_DOCUMENT_DIRECTORY or not os.path.exists(BASE_DOCUMENT_DIRECTORY):
        print(f"Error: BASE_DOCUMENT_DIRECTORY ('{BASE_DOCUMENT_DIRECTORY}') is not set or does not exist.")
        return None

    version_folder_name = f"out_llm_{processing_version}"
    filepath = os.path.join(BASE_DOCUMENT_DIRECTORY, version_folder_name, filename)

    if not os.path.exists(filepath):
        print(f"Error: Document file '{filepath}' not found.")
        # Fallback: search for the filename in any "out_llm_*" folder if exact version not found
        # This might be too broad but can be a fallback if processing_version metadata is imperfect
        found_alternative = False
        for item in os.listdir(BASE_DOCUMENT_DIRECTORY):
            if item.startswith("out_llm_"):
                alt_path = os.path.join(BASE_DOCUMENT_DIRECTORY, item, filename)
                if os.path.exists(alt_path):
                    print(f"Found '{filename}' in alternative folder: {item}. Using this.")
                    filepath = alt_path
                    found_alternative = True
                    break
        if not found_alternative:
            return None
            
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading document file '{filepath}': {e}")
        return None

def analyze_query_with_llm(user_query, client=None, model="microsoft/phi-4"):
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
    
    if client:
        response_text = query_llm(client, model, prompt)
        try:
            json_start = response_text.find('{'); json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_start < json_end:
                json_str = response_text[json_start:json_end+1]; result = json.loads(json_str)
                for key in ["quien", "cuando", "donde", "que"]:
                    if key not in result or not isinstance(result[key], list): result[key] = []
                if "consulta_refinada" not in result or not isinstance(result["consulta_refinada"], str):
                    result["consulta_refinada"] = user_query
                return result
            else: raise json.JSONDecodeError("No JSON object found", response_text, 0)
        except json.JSONDecodeError as e:
            print(f"LLM JSON parse error: {e}. Response: {response_text}")
            return {"quien": [], "cuando": [], "donde": [], "que": [user_query], "consulta_refinada": user_query}
    else:
        return {"quien": [], "cuando": [], "donde": [], "que": [user_query], "consulta_refinada": user_query}

def generate_answer_with_llm(user_query, full_doc_context, chunk_contexts, client=None, model="microsoft/phi-4"):
    contexts_for_prompt = []
    if full_doc_context and full_doc_context.get('text'):
        contexts_for_prompt.append(
            f"--- DOCUMENTO COMPLETO PRINCIPAL (Archivo: {full_doc_context.get('filename', 'Desconocido')}, Versión de procesamiento: {full_doc_context.get('processing_version', 'N/A')}) ---\n"
            f"{full_doc_context['text']}\n"
            f"--- FIN DEL DOCUMENTO COMPLETO PRINCIPAL ---"
        )
    
    if chunk_contexts:
        chunk_intro = "\n\n--- FRAGMENTOS ADICIONALES DE OTROS DOCUMENTOS ---\n"
        contexts_for_prompt.append(chunk_intro)
        for i, chunk_ctx in enumerate(chunk_contexts): # chunk_ctx is a passage_data dict
            contexts_for_prompt.append(
                f"Fragmento {i+1} (Del Archivo: {chunk_ctx.get('base_document_id', 'Desconocido')}, Versión: {chunk_ctx.get('processing_version', 'N/A')}):\n"
                f"{chunk_ctx['text']}\n---"
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
    if client:
        return query_llm(client, model, prompt)
    else:
        return f"Respuesta de muestra para '{user_query}' basada en {'documento completo' if full_doc_context else ''} y {len(chunk_contexts)} fragmentos."

def manage_search_terms(analyzed_query_dict):
    search_terms = []
    refined_query = analyzed_query_dict.get("consulta_refinada")
    if refined_query and isinstance(refined_query, str) and refined_query.strip(): search_terms.append(refined_query.strip())
    for key_type in ["quien", "cuando", "donde", "que"]:
        keywords = analyzed_query_dict.get(key_type, []);
        if isinstance(keywords, list): search_terms.extend(k for k in keywords if isinstance(k, str) and k.strip())
        elif isinstance(keywords, str) and keywords.strip(): search_terms.append(keywords.strip())
    return sorted(list(set(search_terms))) if search_terms else []


def chat_search(message, history):
    global passages_data, bi_encoder_model, cross_encoder_model, corpus_embeddings_tensor, llm_client_instance, knowledge_graph_instance

    # Accumulator for the response string
    response_parts = []

    if not all([passages_data, bi_encoder_model, cross_encoder_model, corpus_embeddings_tensor is not None]):
        response_parts.append("Error: Modelos o datos no inicializados. Revisa la consola.")
        yield "\n".join(response_parts)
        return

    user_query = message.strip()
    if not user_query:
        response_parts.append("Introduce una consulta.")
        yield "\n".join(response_parts)
        return

    # --- Stage 1: Query Analysis and "Buscando" line ---
    analyzed_query_dict = analyze_query_with_llm(user_query, client=llm_client_instance)
    user_query_mejorado = analyzed_query_dict.get("consulta_refinada", user_query)
    if not user_query_mejorado.strip(): user_query_mejorado = user_query
    
    keywords_display_list = sorted(list(set(k for cat in ["quien","cuando","donde","que"] for k in analyzed_query_dict.get(cat, []) if isinstance(k, str) and k.strip())))
    keywords_display_str = ", ".join(keywords_display_list) if keywords_display_list else "ninguna palabra clave específica"
    
    response_parts.append(f"Buscando: {{{user_query_mejorado}, [{keywords_display_str}]}}\n")
    response_parts.append("") # Add a blank line

    # --- Stage 2: Retrieval ---
    search_terms = manage_search_terms(analyzed_query_dict)
    if not search_terms: search_terms = [user_query_mejorado]

    all_candidate_chunks_scored = []
    MAX_SEARCH_TERMS = 5; BI_ENCODER_TOP_K = 20; CROSS_ENCODER_THRESHOLD = 0.0

    for term in search_terms[:MAX_SEARCH_TERMS]:
        q_embedding = bi_encoder_model.encode(term, convert_to_tensor=True)
        # Ensure corpus_embeddings_tensor is on the same device
        if q_embedding.device != corpus_embeddings_tensor.device:
            corpus_tensor_device_equiv = corpus_embeddings_tensor.to(q_embedding.device)
        else:
            corpus_tensor_device_equiv = corpus_embeddings_tensor

        bi_hits = util.semantic_search(q_embedding, corpus_tensor_device_equiv, top_k=BI_ENCODER_TOP_K)
        if not bi_hits or not bi_hits[0]: continue

        cross_input = [[term, passages_data[hit['corpus_id']]['text']] for hit in bi_hits[0]]
        cross_scores = cross_encoder_model.predict(cross_input, show_progress_bar=False)

        for i, hit in enumerate(bi_hits[0]):
            if cross_scores[i] > CROSS_ENCODER_THRESHOLD:
                passage_info = passages_data[hit['corpus_id']]
                all_candidate_chunks_scored.append({
                    'passage_data': passage_info,
                    'cross_score': float(cross_scores[i]),
                    'corpus_id': hit['corpus_id']
                })
    
    all_candidate_chunks_scored.sort(key=lambda x: x['cross_score'], reverse=True)

    # --- Stage 3: Context Selection for LLM ---
    MAX_ADDITIONAL_CHUNKS_FOR_LLM = 3
    full_document_for_llm_data = None
    additional_chunks_for_llm_data = []
    conceptual_doc_scores = {} 
    
    for scored_chunk in all_candidate_chunks_scored:
        passage_details = scored_chunk['passage_data']
        conceptual_id = passage_details['conceptual_doc_id']
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
        full_text = load_full_document_by_details(fname, p_version)
        if full_text:
            full_document_for_llm_data = {'filename': fname, 'processing_version': p_version, 'text': full_text}
            print(f"Loaded full text: {fname} (V: {p_version})")
        else: print(f"Failed to load full text: {fname} (V: {p_version})")

    used_corpus_ids_for_additional_llm_chunks = set()
    if full_document_for_llm_data:
         used_corpus_ids_for_additional_llm_chunks.add(sorted_conceptual_docs[0][1][3])

    for scored_chunk in all_candidate_chunks_scored:
        if len(additional_chunks_for_llm_data) >= MAX_ADDITIONAL_CHUNKS_FOR_LLM: break
        passage_details = scored_chunk['passage_data']
        chunk_corpus_id = scored_chunk['corpus_id']
        is_diff_concept = top_conceptual_id_for_full_doc is None or \
                          passage_details['conceptual_doc_id'] != top_conceptual_id_for_full_doc
        if is_diff_concept and chunk_corpus_id not in used_corpus_ids_for_additional_llm_chunks:
            additional_chunks_for_llm_data.append(passage_details)
            used_corpus_ids_for_additional_llm_chunks.add(chunk_corpus_id)
            
    # --- Stage 4: Generate LLM Answer ---
    llm_answer = generate_answer_with_llm(
        user_query_mejorado,
        full_document_for_llm_data,
        additional_chunks_for_llm_data,
        client=llm_client_instance
    )
    response_parts.append(f"Respuesta: {llm_answer}\n")

    # --- Stage 5: Full Transparency Output ---
    response_parts.append("--- CONTEXTO UTILIZADO PARA GENERAR LA RESPUESTA ---")
    
    if full_document_for_llm_data:
        response_parts.append("\nDOCUMENTO COMPLETO PRINCIPAL:")
        response_parts.append(f"  Archivo: {full_document_for_llm_data['filename']}")
        response_parts.append(f"  Versión de Procesamiento: {full_document_for_llm_data['processing_version']}")        
        response_parts.append(f"  Contenido:\n{full_document_for_llm_data['text']}")
    else:
        response_parts.append("\nNo se utilizó un documento completo principal.")

    if additional_chunks_for_llm_data:
        response_parts.append("\nFRAGMENTOS ADICIONALES DE OTROS DOCUMENTOS:\n")
        for i, chunk_data in enumerate(additional_chunks_for_llm_data):
            response_parts.append(f"  \nFragmento Adicional {i+1}:")
            response_parts.append(f"    Archivo: {chunk_data.get('base_document_id', 'N/A')}")
            response_parts.append(f"    Versión de Procesamiento: {chunk_data.get('processing_version', 'N/A')}")
            response_parts.append(f"    Texto del Fragmento:\n{chunk_data.get('text', '')}")
    else:
        response_parts.append("\nNo se utilizaron fragmentos adicionales de otros documentos.")
    
    if not full_document_for_llm_data and not additional_chunks_for_llm_data:
        response_parts.append("\n(No se proporcionó contexto específico al LLM para esta respuesta, o la recuperación falló).")

    # Yield the entire accumulated response at once
    yield "\n".join(response_parts)

if __name__ == "__main__":
    print("Starting RAG program initialization...")
    if not os.path.exists(BASE_DOCUMENT_DIRECTORY) and BASE_DOCUMENT_DIRECTORY == "":
        print("Couldn't access base directory or doesn't exist.")

    initialize_models_and_data()


    #TODO; change html
    html_content = "hi"
    html_path = "/home/pfont/rag/graph_demo_files/my_interactive_kg.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()


    if not all([passages_data, bi_encoder_model, cross_encoder_model, corpus_embeddings_tensor is not None]):
        print("Critical error during initialization. Exiting.")
    else:
        print("Models and data loaded. Launching Gradio interface...")
        chatbot = gr.ChatInterface(
            fn=chat_search,
            title="Chatbot Histórico",
            description="Pregunta sobre documentos históricos. Se mostrará todo el contexto usado para la respuesta.",
            theme="default",
            examples=[
                "¿Qué sentencias se dictaron en agosto de 1939?",
                "¿Qué le ocurrió a DON PEDRO MANUEL RUIZ PÁRDO durante la guerra?",
                "¿Quiénes fueron acusados de auxilio a la rebelión?"
            ],
            cache_examples=False 
        )

        with gr.Blocks() as demo:
            with gr.Row():
                chatbot.render()
                gr.HTML(html_content)

        demo.launch()