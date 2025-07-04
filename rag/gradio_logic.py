import gradio as gr
from gradio_modal import Modal
from datetime import datetime
import os
import uuid 
import time 
from rag_core import RAGSystem
from pathlib import Path
import html
from graph_search import get_centrality_measures


print("Starting RAG program initialization...")
rag_system = RAGSystem()
rag_system.initialize_models_and_data()
HTML_HEIGHT=600

os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0" #TODO: Check, doesn't work
INPUT_FOLDER = "/data/users/pfont/input"

def html_text_context(text):
    # We convert some characters to HTML to improve visualization
    context_text = html.escape(text)
    context = context_text.replace('*', '&#42;') 
    context = context.replace('-', '&#45;')   
    context = context.replace('\n', '&#10;')
    return context

def get_images(document_id):
    folder = Path(INPUT_FOLDER)

    list_images = sorted([
        str(file)
        for file in folder.glob(f"rsc37_rsc176_{document_id}_*.jpg")
        if file.is_file()
    ])
    return list_images


def create_new_conversation_entry(base_name):
    #TODO: Change name
    """Helper to create a new conversation dictionary."""
    id_actual = str(uuid.uuid4())
    # Show first 8 characters of ID for readability
    short_id = id_actual[:8]
    return {
        "id": id_actual,
        "name": f"{base_name}_{id_actual}",
        "history": [],
        "graph_path": None
    }

def get_conversation_by_id(conv_id, conversations_list):
    """Helper to find a conversation in the list by its ID."""
    for conv in conversations_list:
        if conv["id"] == conv_id:
            return conv
    return None

# Helper function to create Gradio accessible file URLs
def make_gradio_file_url(file_path):
    if not file_path or not os.path.exists(file_path):
        return None
    abs_path = os.path.abspath(file_path)
    # Appending a timestamp can help avoid caching issues with Gradio file serving
    return f"/gradio_api/file={abs_path}?v={time.time()}"


# Returns html from a path to an html (GRAPH)
def get_graph_html(graph_file_path):
    """Return HTML content for displaying the graph"""
    if not graph_file_path or not os.path.exists(graph_file_path):
        return "<p>No hay grafo disponible.</p>"

    abs_path = os.path.abspath(graph_file_path)
    try:
        import time
        iframe_src = f"/gradio_api/file={abs_path}?v={time.time()}"
        return f'<iframe src="{iframe_src}" width="100%" height="{HTML_HEIGHT}px" style="border:none;" sandbox="allow-scripts allow-same-origin"></iframe>'
    except Exception as e:
        return f"<p>Error al cargar el grafo: {str(e)}</p>"

# Gradio UI
with gr.Blocks(title="ArchiText", theme='default') as demo:
    # --- State Variables ---
    initial_conversation = create_new_conversation_entry("Conversación") # Crea nueva conversación
    all_conversations_state = gr.State([initial_conversation])
    active_conversation_id_state = gr.State(initial_conversation["id"])
    showing_global_graph_state = gr.State(False) # False = current graph, True = global graph
    gr.Markdown(
        "<h1 style='text-align: center; font-size: 3em;'>ArchiText</h1>\n<h2 style='text-align: center; font-size: 1.5em;'>Conversational Retrieval and Knowledge Modeling from Historical Documents</h2>",
        elem_id="title"
    )
    current_conv_id_display = gr.Markdown(
        f"<div style='text-align: center;'><strong>ID de sesión:</strong> <code onclick='navigator.clipboard.writeText(\"{rag_system.session_id}\");' style='cursor: pointer;'>{rag_system.session_id}</code></div>",
        elem_id="session_id"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=250): # Sidebar for conversations
            gr.Markdown(
                "<h3 style='text-align: center; color: #4CAF50;'>Conversaciones</h3>")
            current_conv_id_display = gr.Markdown(
                f"<div style='text-align: center;'><strong>ID Actual:</strong> <code onclick='navigator.clipboard.writeText(\"{initial_conversation['id']}\");' style='cursor: pointer;'>{initial_conversation['id']}</code></div>",
                elem_id="current_conv_id"
            )
            conversation_selector = gr.Radio(
                label="Selecciona una conversación",
                choices=[(initial_conversation["name"], initial_conversation["id"])],
                value=initial_conversation["id"],
                type="value"
            )
            new_conv_btn = gr.Button("+ Nueva Conversación")

        with gr.Column(scale=3): 
            chatbot = gr.Chatbot(
                label="Chat", 
                show_label=False,
                height=HTML_HEIGHT,
                type="tuples"
            )
            
            with gr.Group():
                with gr.Row(equal_height=True):
                    msg_input = gr.Textbox(
                        placeholder="Pregunta sobre personas, sitios y lugares relacionados con la Guerra Civil Española",
                        label="Tu Pregunta",
                        show_label=False,
                        lines=2,
                        interactive=True  # Ensure the textbox is interactive
                    )
                    send_btn = gr.Button("Shift ↵", elem_id="send_button", variant="primary", size="lg", scale=0)

        with gr.Column(scale=2, min_width=300): # Graph display area
            gr.Markdown(
                "<h3 style='text-align: center; color: #4CAF50;'>Grafo de Conocimiento</h3>")
            toggle_graph_btn = gr.Button("Ver Grafo Global", scale=1)
            graph_output = gr.HTML()


    # --- Event Handler Functions ---
    """
    Changes graph view (current graph or global graph) (changes the button and graph accordingly)
    """
    def handle_toggle_graph_view(is_currently_showing_global, active_conv_id, all_conversations):
        new_showing_global = not is_currently_showing_global
        button_text = "Ver Grafo Actual" if new_showing_global else "Ver Grafo Global"
        
        graph_html_content = ""
        if new_showing_global: #If we change to showing global, show global
            graph_html_content = get_graph_html(rag_system.session_global_graph_html_path)
            centrality_texts = get_centrality_measures(rag_system.knowledge_graph_global)

            centrality_info_html = "<div style='margin-top: 15px; padding: 10px; border: 1px solid #e0e0e0; background-color: #f9f9f9; border-radius: 5px;'>"
            if centrality_texts:
                centrality_info_html += "<h4 style='margin-top:0; margin-bottom: 8px; color: #333;'>Métricas de Centralidad del Grafo Global:</h4><ul style='color: #333; list-style-type: disc; margin-left: 20px;'>"
                centrality_info_html += "".join(centrality_texts)
                centrality_info_html += "</ul>"
            elif not rag_system.knowledge_graph_global or not rag_system.knowledge_graph_global.nodes:
                    centrality_info_html += "<p style='margin:0; color: #333;'><i>El grafo global está vacío o no contiene nodos para calcular métricas de centralidad.</i></p>"
            else:
                centrality_info_html += "<p style='margin:0; color: #333;'><i>No se encontraron entidades específicas (personas/ubicaciones) o datos suficientes para calcular todas las métricas de centralidad en el grafo global.</i></p>"
            centrality_info_html += "</div>"
            
            graph_html_content += centrality_info_html
        else:
            active_conv = get_conversation_by_id(active_conv_id, all_conversations)
            if active_conv:
                graph_html_content = get_graph_html(active_conv["graph_path"]) #we get graph path and then html
            else: # Fallback if no active conversation somehow
                graph_html_content = get_graph_html(None) #empty in case of errors
                
        return graph_html_content, gr.update(value=button_text), new_showing_global

    def handle_new_conversation_click(current_conversations):
        new_conv_name = f"Conversación"
        new_conv = create_new_conversation_entry(new_conv_name)
        updated_conversations = current_conversations + [new_conv]
        
        choices = [(c["name"], c["id"]) for c in updated_conversations]
        
        return (
            [],  # New chat history (empty for chatbot)
            get_graph_html(None),  # New graph (empty for graph_output)
            gr.update(choices=choices, value=new_conv["id"]), # Update conversation_selector
            updated_conversations, # Update all_conversations_state
            new_conv["id"],  # Update active_conversation_id_state
            gr.update(value="Ver Grafo Global"), # Reset toggle button text
            False, # Reset showing_global_graph_state
            gr.update(value=f"**ID Actual:** `{new_conv['id']}`") # Update current conv ID display
        )

    def handle_switch_conversation_change(selected_conv_id, current_conversations):
        active_conv_to_switch_to = get_conversation_by_id(selected_conv_id, current_conversations)
        
        if not selected_conv_id and current_conversations: # Should not happen if radio always has a value
            active_conv_to_switch_to = current_conversations[0]
            selected_conv_id = active_conv_to_switch_to["id"]
        
        if active_conv_to_switch_to:
            return (
                active_conv_to_switch_to["history"],
                get_graph_html(active_conv_to_switch_to["graph_path"]),
                active_conv_to_switch_to["id"],
                gr.update(value="Ver Grafo Global"), # Reset toggle button text
                False, # Reset showing_global_graph_state
                gr.update(value=f"**ID Actual:** `{active_conv_to_switch_to['id']}`") # Update current conv ID display
            )
        
        # Fallback
        return [], get_graph_html(None), selected_conv_id, gr.update(value="Ver Grafo Global"), False, gr.update(value=f"**ID Actual:** `{selected_conv_id if selected_conv_id else 'N/A'}...`")

    def on_send_message_submit(user_input, active_conv_id, current_conversations_list_state):
        # This function is a generator
        active_conv_index = -1
        for i, c in enumerate(current_conversations_list_state):
            if c["id"] == active_conv_id:
                active_conv_index = i
                break
        
        if active_conv_index == -1:
            print(f"Error: Active conversation ID '{active_conv_id}' not found during send message.")
            # Try to find a fallback or yield current state to avoid crashing
            current_chat_history = []
            current_graph_html = get_graph_html(None)
            if current_conversations_list_state:
                fallback_conv = current_conversations_list_state[0]
                current_chat_history = fallback_conv["history"]
                current_graph_html = get_graph_html(fallback_conv["graph_path"])

            yield current_chat_history, current_graph_html, user_input, current_conversations_list_state, gr.update(value="Ver Grafo Global"), False
            return

        # Create a new list of conversations to ensure Gradio detects state change
        updated_conversations_list = [conv.copy() for conv in current_conversations_list_state]
        active_conv = updated_conversations_list[active_conv_index]

        if not user_input.strip():
            yield active_conv["history"], get_graph_html(active_conv["graph_path"]), "", updated_conversations_list, gr.update(value="Ver Grafo Global"), False
            return

        active_conv["history"] = active_conv["history"] + [[user_input, None]]
        
        # Yield intermediate state: user message shown, bot thinking
        # Graph remains current conversation's graph, toggle button reset
        yield active_conv["history"], get_graph_html(active_conv["graph_path"]), "", updated_conversations_list, gr.update(value="Ver Grafo Global"), False
        
        try:
            answer_info, new_graph_path = rag_system.chat_search(user_input, active_conv["history"], conversation_id=active_conv_id)
            #To recuperate images and show
            context_text = answer_info.get("text", None) 

            # Construir la respuesta enriquecida en Markdown
            markdown_response_parts = ["".join(answer_info['llm_answer'])]
           
            if context_text:
                markdown_response_parts.append(
                    f'\n\n**Contexto:**\n <pre style="max-height: 200px; overflow-y: auto; border: 1px solid #ccc; padding: 8px; background-color: #333; white-space: pre-line;">{html_text_context(context_text)}</pre>'
                )

                markdown_response_parts.append(f"\n*Contexto recuperado del documento {answer_info['id']} utilizando la versión {answer_info['processing_version']}*")
                
                image_doc_paths = get_images(str(answer_info["id"]))
                
                if image_doc_paths and len(image_doc_paths) > 0:
                    markdown_response_parts.append("\n\n**Documento:**")
                    markdown_response_parts.append('<div style="display: flex; flex-wrap: wrap; gap: 4px; justify-content: flex-start; align-items: center; max-width: 100%; overflow-x: auto;">')  # Start a flex container for images
                    for i, img_path in enumerate(image_doc_paths, start=1):
                        img_url = make_gradio_file_url(img_path)
                        if img_url:
                            # Mostrar todas las imágenes del documento en línea con enlaces para abrir en otra pestaña
                            markdown_response_parts.append(
                                f'''<a href="{img_url}" target="_blank" style="flex-shrink: 0;"><img onmouseover="this.style.transform='scale(4)'" onmouseout="this.style.transform='scale(1)'" src="{img_url}" alt="Página {i}" style="width: 80px; height: auto; object-fit: contain; display: block;"/></a>''')

                    markdown_response_parts.append("</div>")  # Close the flex container
            
            if rag_system.GRAPH_ANSWER:
                graph_llm_answer_text = answer_info.get("graph_answer")
                graph_context_list = answer_info.get("graph_context") # This is a list of strings

                if graph_llm_answer_text:
                    markdown_response_parts.append(f"\n<hr style='border-top: 1px dashed #ccc;'>") # Visual separator
                    markdown_response_parts.append(f"\n\n**Respuesta (basada en grafo global):**\n {html.escape(graph_llm_answer_text)}")
                
                if graph_context_list:
                    # Join the list of context strings into a single block for display
                    graph_context_str_display = "---".join(graph_context_list) # Separator between contexts
                    markdown_response_parts.append(f"\n\n**Contexto del grafo global:**\n <pre style='max-height: 200px; overflow-y: auto; border: 1px solid #ccc; padding: 8px; background-color: #333; white-space: pre-line;'>{html_text_context(graph_context_str_display)}</pre>")
            
            bot_response = "".join(markdown_response_parts)

        except Exception as e:
            bot_response = f"Error al procesar la consulta: {str(e)}"
            new_graph_path = active_conv["graph_path"] # Keep old graph on error
            print(f"Error during RAG search: {e}")

        active_conv["history"][-1][1] = bot_response
        active_conv["graph_path"] = new_graph_path
        
        # Final yield: bot response, updated graph for current conversation, toggle button reset
        yield active_conv["history"], get_graph_html(active_conv["graph_path"]), "", updated_conversations_list, gr.update(value="Ver Grafo Global"), False


    def load_initial_ui(active_id, conversations):
        conv = get_conversation_by_id(active_id, conversations)
        initial_history = []
        initial_graph_html = get_graph_html(None)
        if conv:
            initial_history = conv["history"]
            initial_graph_html = get_graph_html(conv["graph_path"])
        return initial_history, initial_graph_html, "Ver Grafo Global", False # Initial button text and toggle state

    # --- Event Wiring ---
    demo.load(
        fn=load_initial_ui,
        inputs=[active_conversation_id_state, all_conversations_state],
        outputs=[chatbot, graph_output, toggle_graph_btn, showing_global_graph_state]
    )

    new_conv_btn.click(
        fn=handle_new_conversation_click,
        inputs=[all_conversations_state],
        outputs=[chatbot, graph_output, conversation_selector, all_conversations_state, active_conversation_id_state, toggle_graph_btn, showing_global_graph_state, current_conv_id_display]
    )

    conversation_selector.change(
        fn=handle_switch_conversation_change,
        inputs=[conversation_selector, all_conversations_state],
        outputs=[chatbot, graph_output, active_conversation_id_state, toggle_graph_btn, showing_global_graph_state, current_conv_id_display]
    )
    
    # For on_send_message_submit, inputs now include showing_global_graph_state,
    # and outputs include toggle_graph_btn and showing_global_graph_state.
    # However, on_send_message_submit will always reset to current graph view.
    # The toggle button is the explicit way to view global.
    # Remove the send button and use the submit event of the textbox for sending messages
    msg_input.submit(
        fn=on_send_message_submit,
        inputs=[msg_input, active_conversation_id_state, all_conversations_state],
        outputs=[chatbot, graph_output, msg_input, all_conversations_state, toggle_graph_btn, showing_global_graph_state]
    )
    
    send_btn.click(
        fn=on_send_message_submit,
        inputs=[msg_input, active_conversation_id_state, all_conversations_state],
        outputs=[chatbot, graph_output, msg_input, all_conversations_state, toggle_graph_btn, showing_global_graph_state]
    )

    toggle_graph_btn.click(
        fn=handle_toggle_graph_view,
        inputs=[showing_global_graph_state, active_conversation_id_state, all_conversations_state],
        outputs=[graph_output, toggle_graph_btn, showing_global_graph_state]
    )
# Launch app
if __name__ == "__main__":
    graph_doc_dir = rag_system.GRAPH_DOCUMENT_DIRECTORY
    allowed_paths_list = []
    if graph_doc_dir:
        if not os.path.exists(graph_doc_dir):
            os.makedirs(graph_doc_dir, exist_ok=True)
        allowed_paths_list.append(graph_doc_dir)
        print(f"Graph document directory '{graph_doc_dir}' will be allowed for serving files (including session global graphs).")
    else:
        print("Warning: rag_system.GRAPH_DOCUMENT_DIRECTORY is not set. Graphs may not be served correctly.")


    # If GLOBAL_GRAPH_PATH is outside graph_doc_dir and needs its own directory to be allowed:
    # global_graph_dir = os.path.dirname(GLOBAL_GRAPH_PATH)
    # if global_graph_dir not in allowed_paths_list and os.path.isdir(global_graph_dir):
    # allowed_paths_list.append(global_graph_dir)
    allowed_paths_list.append("/data/users/pfont/final_documents")
    allowed_paths_list.append("/data/users/pfont/input")
    print(f"Launching Gradio app. Allowed paths for graphs: {allowed_paths_list}")
    demo.launch(allowed_paths=allowed_paths_list, share=False)