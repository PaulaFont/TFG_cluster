from openai import OpenAI
from llm_utils import *
from pre_processing import *
import torch 
import json
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
import pandas as pd
import torch
import json
import time # For Gradio streaming simulation if used
import gradio as gr # For the UI


passages_data = [] # This will now be a list of dicts
bi_encoder_model = None
cross_encoder_model = None
corpus_embeddings_tensor = None
llm_client_instance = None

final_answer = ""

def initialize_models_and_data(llm_model_name="microsoft/phi-4", bi_encoder_name="msmarco-bert-base-dot-v5", cross_encoder_name='cross-encoder/ms-marco-MiniLM-L6-v2', passages_filename= "passages_data.json"):
    global passages_data, bi_encoder_model, cross_encoder_model, corpus_embeddings_tensor, llm_client_instance

    if not start_llm_server(llm_model_name, port=8000): # Returns an error
        llm_client_instance = None
        return
    else:
        llm_client_instance = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123"
        )
    
    print("Creating passages and metadata...")
    if not os.path.exists(passages_filename):
        print(f"File {passages_filename} not found. Creating new passages...")
        df_passages = create_passages()
        passages_data = df_passages.to_dict(orient='records') # List of dicts
        # Save to file
        with open("passages_data.json", 'w') as f:
            json.dump(passages_data, f, indent=2)
        
        print(f"Saved {len(passages_data)} passages to {passages_filename}")
    
    # Load from JSON file
    with open(passages_filename, 'r') as f:
        passages_data = json.load(f)   

    if not passages_data:
        print("No passages were created. Check your data path and processing logic.")
        return

    passage_texts = [p['text'] for p in passages_data]

    print("Loading embedding models...")
    try:
        bi_encoder_model = SentenceTransformer(bi_encoder_name)
        cross_encoder_model = CrossEncoder(cross_encoder_name)
    except Exception as e:
        print(f"Error loading sentence-transformer models: {e}")
        print("Please ensure the models are correctly specified and accessible.")
        print(f"Attempted Bi-Encoder: {bi_encoder_name}")
        print(f"Attempted Cross-Encoder: {cross_encoder_name}")
        return

    print("Creating embeddings...")
    corpus_embeddings_tensor = bi_encoder_model.encode(passage_texts[0:10], convert_to_tensor=True, show_progress_bar=True) #TODO:change
    print("Initialization complete.")

"""Returns:
        dict: A dictionary containing the extracted entities and a refined search query
"""
def analyze_query_with_llm(user_query, client=None, model="microsoft/phi-4"):
    """
    Uses an LLM to extract key entities from a Spanish language query,
    identifying the WHO (people), WHEN (dates/time periods), WHERE (locations),
    and WHAT (events, topics, or key terms).
    
    Args:
        user_query (str): The user's question in Spanish
        client: The LLM client object
        model (str): Name of the model to use
        
    Returns:
        dict: A dictionary containing the extracted entities and a refined search query
    """
    # Create a prompt in Spanish that instructs the LLM to identify key entities
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

    # Call the LLM with the prompt
    if client:
        response = query_llm(client, model, prompt)
        
        # Parse the JSON response
        # In a production environment, you'd want to add error handling here
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # Fallback - could implement more robust parsing if the LLM 
            # doesn't consistently return valid JSON
            return {
                "quien": [],
                "cuando": [],
                "donde": [],
                "que": [user_query],
                "consulta_refinada": user_query
            }
    else:
        # If no client is provided, return a placeholder
        print("No LLM client provided, returning placeholder analysis")
        return {
            "quien": [],
            "cuando": [],
            "donde": [],
            "que": [user_query],
            "consulta_refinada": user_query
        }


def generate_answer_with_llm(user_query, contexts, client=None, model="microsoft/phi-4"):
    """
    Genera una respuesta utilizando un LLM basado en la consulta del usuario y los contextos recuperados.
    Diseñado específicamente para documentos históricos en español.
    
    Args:
        user_query (str): La pregunta del usuario en español
        contexts (list): Lista de diccionarios, cada uno con información de un pasaje recuperado
        client: El cliente LLM para realizar la consulta
        model (str): Nombre del modelo a utilizar
        
    Returns:
        str: La respuesta generada por el LLM basada en los contextos proporcionados
    """
    # Primero, verifiquemos si hay contextos disponibles
    if not contexts or len(contexts) == 0:
        return "No encontré información relevante en los documentos para responder tu pregunta."
    
    # Formateamos los contextos en un formato legible para el LLM
    context_str = "\n\n---\n\n".join([
        f"Documento Fuente: {ctx['base_document_id']} "
        f"(Versión: {ctx['processing_version']}, "
        f"Contenido: {ctx['text']}" 
        for ctx in contexts
    ])
    
    # Creamos el prompt en español que instruye al LLM sobre cómo responder
    prompt = f"""
    Eres un asistente de IA experto en documentos históricos en español.

    Responde a la siguiente pregunta del usuario basándote *únicamente* en los contextos proporcionados de los documentos históricos.
    Sé conciso y responde directamente a la pregunta.

    No inventes información. Si el contexto no proporciona una respuesta, indícalo claramente.

    Pregunta del Usuario: "{user_query}"

    Contextos Proporcionados:
    {context_str}

    Respuesta:
    """

    # Llamada al LLM con el prompt
    if client:
        response = query_llm(client, model, prompt)
        return response
    else:
        # Si no hay cliente LLM disponible, proporcionamos una respuesta de muestra
        print("No se proporcionó cliente LLM, devolviendo respuesta de muestra")
        
        # Generamos una respuesta de muestra basada en los contextos disponibles
        sample_answer = f"En base a los documentos proporcionados sobre tu pregunta: '{user_query}'\n\n"
        
        # Limitamos a mostrar información de los primeros 2 contextos por brevedad
        for i, ctx in enumerate(contexts[:2]):
            doc_id = ctx['base_document_id']
            version = ctx['processing_version']
            section = ctx.get('section_type', 'N/A')
            text_snippet = ctx['text'][:150] + "..." if len(ctx['text']) > 150 else ctx['text']
            
            sample_answer += f"- Según el documento '{doc_id}' (Versión: {version}, Sección: {section}):\n"
            sample_answer += f"  {text_snippet}\n\n"
        
        sample_answer += "\nEn un sistema real, un LLM sintetizaría esta información en una respuesta coherente y citaría apropiadamente las fuentes."
        return sample_answer

def manage_consulta_refinada(analyzed_query_dict):
    llista_consultas = []
    for value in analyzed_query_dict.values():
        llista_consultas.extend(value)
    return llista_consultas
    
def add_final_answer(text):
    global final_answer
    final_answer += f"\n{text}\n"

def chat_search(message, history):
    global passages_data, bi_encoder_model, cross_encoder_model, corpus_embeddings_tensor, llm_client_instance, final_answer

    if not all([passages_data, bi_encoder_model, cross_encoder_model, corpus_embeddings_tensor is not None, llm_client_instance]):
        yield "Models or data not initialized. Please check the console."
        return

    user_query = message
    yield (f"User query: {user_query}")

    # 1. Query Understanding Agent
    analyzed_query_dict = analyze_query_with_llm(user_query, client=llm_client_instance)
    consultas = manage_consulta_refinada(analyzed_query_dict)
    yield("Consultas: ")
    for c in consultas:
        yield(c)

    all_contexts = []
    return #TODO: change

    # 2. Retrieval (Bi-Encoder)
    for consulta in consultas:
        print(f"Searching for: {consulta}")
        add_final_answer(f"Searching for: {consulta}")
        question_embedding = bi_encoder_model.encode(consulta, convert_to_tensor=True)
        if torch.cuda.is_available(): # Check if CUDA is available
            question_embedding = question_embedding.cuda()

        # Make sure corpus_embeddings_tensor is on the same device as question_embedding
        # If corpus_embeddings_tensor was created on CPU and question_embedding moved to CUDA:
        if question_embedding.device != corpus_embeddings_tensor.device:
            corpus_embeddings_tensor_device = corpus_embeddings_tensor.to(question_embedding.device)
        else:
            corpus_embeddings_tensor_device = corpus_embeddings_tensor

        # Semantic search
        # Increase K for bi-encoder to give more candidates to cross-encoder
        retrieved_hits = util.semantic_search(question_embedding, corpus_embeddings_tensor_device, top_k=20)[0]


        # 3. Re-ranking (Cross-Encoder)
        if not retrieved_hits:
            yield "No initial results found by the bi-encoder." # Changed from direct yield
            # For LLM generation, we'd pass this to the generate_answer_with_llm
            final_answer = generate_answer_with_llm(user_query, [], client=llm_client_instance)
            yield final_answer
            return

        cross_inp = [[consulta, passages_data[hit['corpus_id']]['text']] for hit in retrieved_hits]
        cross_scores = cross_encoder_model.predict(cross_inp)
        add_final_answer(f"Context trobat per la consulta {consulta}:")
        
        # Attach scores and full passage data to hits
        for i, hit in enumerate(retrieved_hits):
            hit['cross-score'] = cross_scores[i]
            # Store the full passage dictionary (text + metadata)
            hit['passage_data'] = passages_data[hit['corpus_id']]
            add_final_answer(f"== Original Document: {hit['passage_data']['base_document_id']}, Versió: {hit['passage_data']['processing_version']}")
            add_final_answer(hit['passage_data']["text"])

        # Filter by a threshold and sort by cross-encoder score
        # You might want to adjust the threshold or remove it if the LLM handles low-relevance well
        relevant_hits = [hit for hit in retrieved_hits if hit['cross-score'] > -5] # Lowered threshold for more context
        relevant_hits = sorted(relevant_hits, key=lambda x: x['cross-score'], reverse=True)

        # Select top_k for LLM context (e.g., top 5)
        top_k_for_llm = 5 #TODO: vary this number depending on query
        context_for_llm = [hit['passage_data'] for hit in relevant_hits[:top_k_for_llm]]
        all_contexts.extend(context_for_llm)

    # 4. Post-Retrieval / Answer Generation with LLM
    if not context_for_llm:
        add_final_answer(generate_answer_with_llm(user_query, [], client=llm_client_instance))
        yield final_answer
        return
    
    add_final_answer(generate_answer_with_llm(user_query, context_for_llm, client=llm_client_instance))

    yield final_answer


if __name__ == "__main__":
 
    initialize_models_and_data() # Load models and data once at startup

    if not all([
        passages_data is not None,  # Also good practice to be explicit here
        bi_encoder_model is not None,
        cross_encoder_model is not None,
        corpus_embeddings_tensor is not None, # Explicitly check if it's not None
        llm_client_instance is not None
    ]):
        print("Failed to initialize. Exiting.")
    else:
        print("Models and data loaded successfully. Launching Gradio interface...")
        demo = gr.ChatInterface(
            fn=chat_search,
            title="Historical Documents Chatbot (RAG Enhanced)",
            description="Ask something about the historical documents. The system will try to answer using relevant excerpts.",
            theme="default"
        )
        demo.launch()