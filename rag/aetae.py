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
        answer_doc_id = {
            "id" : top_conceptual_id_for_full_doc,
            "context_text" : full_text,
        }
                
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