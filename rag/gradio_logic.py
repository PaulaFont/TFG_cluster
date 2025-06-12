import gradio as gr
from datetime import datetime
import os
import uuid 
import time 
from rag_core import RAGSystem

print("Starting RAG program initialization...")
rag_system = RAGSystem()
rag_system.initialize_models_and_data()
HTML_HEIGHT=800

os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0" #TODO: Check, doesn't work


GLOBAL_GRAPH_FILENAME = "global_knowledge_graph.html"
GLOBAL_GRAPH_PATH = None

if rag_system.GRAPH_DOCUMENT_DIRECTORY and os.path.isdir(rag_system.GRAPH_DOCUMENT_DIRECTORY):
    GLOBAL_GRAPH_PATH = os.path.join(rag_system.GRAPH_DOCUMENT_DIRECTORY, GLOBAL_GRAPH_FILENAME)
    print(f"Global graph path set to: {GLOBAL_GRAPH_PATH}")
else:
    print(f"Warning: rag_system.GRAPH_DOCUMENT_DIRECTORY ('{rag_system.GRAPH_DOCUMENT_DIRECTORY}') is not set or not a directory. Global graph feature might not work if the file isn't found or the path isn't allowed.")
    # If GLOBAL_GRAPH_PATH remains None, get_graph_html will show "No hay grafo disponible."

GLOBAL_GRAPH_PATH = "/data/users/pfont/graph/online_knowledge_graph_alsasua.html" #TODO: It's hardcoded

def create_new_conversation_entry(base_name):
    #TODO: Change name
    """Helper to create a new conversation dictionary."""
    id_actual = str(uuid.uuid4())
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
with gr.Blocks(title="Chatbot Histórico con Grafo", theme='default') as demo:
    # --- State Variables ---
    initial_conversation = create_new_conversation_entry("Conversación") # Crea nueva conversación
    all_conversations_state = gr.State([initial_conversation])
    active_conversation_id_state = gr.State(initial_conversation["id"])
    showing_global_graph_state = gr.State(False) # False = current graph, True = global graph

    gr.Markdown(
        "<h1 style='text-align: center;'>Chatbot Histórico con Visualización de Conocimiento</h1>\n<p style='text-align: center;'>Pregunta sobre documentos históricos.</p>",
        elem_id="title"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=250): # Sidebar for conversations
            gr.Markdown(
                "<h3 style='text-align: center; color: #4CAF50;'>Conversaciones</h3>")
            conversation_selector = gr.Radio(
                label="Selecciona una conversación",
                choices=[(initial_conversation["name"], initial_conversation["id"])],
                value=initial_conversation["id"],
                type="value"
            )
            new_conv_btn = gr.Button("+ Nueva Conversación")

        with gr.Column(scale=3): # Main chat area
            chatbot = gr.Chatbot(
            label="Chat", #TODO: hide label
            #bubble_full_width=False, #TODO: check if affects
            height=HTML_HEIGHT
            )
            with gr.Row():  # Place input and send button in the same row
                msg_input = gr.Textbox(
                    placeholder="Escribe tu pregunta aquí...",
                    label="Tu Pregunta",
                    show_label=False,
                    lines=2,
                    interactive=True  # Ensure the textbox is interactive
                )
                # send_btn = gr.Button("➤", elem_id="send_button", variant="primary", scale=1)  # Smaller scale for button #TODO: Decide to put it or not

        with gr.Column(scale=2, min_width=300): # Graph display area
            with gr.Row():
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
            graph_html_content = get_graph_html(GLOBAL_GRAPH_PATH)
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
            False # Reset showing_global_graph_state
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
                False # Reset showing_global_graph_state
            )
        
        # Fallback
        return [], get_graph_html(None), selected_conv_id, gr.update(value="Ver Grafo Global"), False

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
            bot_response, new_graph_path = rag_system.chat_search(user_input, active_conv["history"], conversation_id=active_conv_id)

            # Construir la respuesta enriquecida en Markdown
            markdown_response_parts = [bot_response]
            context_txt_path = "/data/users/pfont/final_documents/rsc37_rsc176_278_all.txt" #TODO:change, needs to be a return from chat_search
            if context_txt_path:
                txt_url = make_gradio_file_url(context_txt_path)
                if txt_url:
                    markdown_response_parts.append(f'\n\n**Contexto:** <a href="{txt_url}" target="_blank">Ver archivo de texto</a>')
            
            image_doc_paths=["/data/users/pfont/input/rsc37_rsc176_278_0.jpg",
            "/data/users/pfont/input/rsc37_rsc176_278_0.jpg",
            "/data/users/pfont/input/rsc37_rsc176_278_0.jpg"] 
           
            #TODO:change, needs to be a return from chat_search
            # Añadir imagen(es) del documento si existen
            if image_doc_paths and len(image_doc_paths) > 0:
                markdown_response_parts.append("\n\n**Documento:**")
                markdown_response_parts.append('<div style="display: flex; flex-wrap: wrap; gap: 4px; justify-content: flex-start; align-items: center; max-width: 100%; overflow-x: auto;">')  # Start a flex container for images
                for i, img_path in enumerate(image_doc_paths, start=1):
                    img_url = make_gradio_file_url(img_path)
                    if img_url:
                        # Mostrar todas las imágenes del documento en línea con enlaces para abrir en otra pestaña
                        markdown_response_parts.append(
                            f'<a href="{img_url}" target="_blank" style="flex-shrink: 0;"><img src="{img_url}" alt="Página {i}" style="width: 80px; height: auto; object-fit: cover; display: block;"/></a>')
                markdown_response_parts.append("</div>")  # Close the flex container
            
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
        outputs=[chatbot, graph_output, conversation_selector, all_conversations_state, active_conversation_id_state, toggle_graph_btn, showing_global_graph_state]
    )

    conversation_selector.change(
        fn=handle_switch_conversation_change,
        inputs=[conversation_selector, all_conversations_state],
        outputs=[chatbot, graph_output, active_conversation_id_state, toggle_graph_btn, showing_global_graph_state]
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
    
    # send_btn.click(
    #     fn=on_send_message_submit,
    #     inputs=[msg_input, active_conversation_id_state, all_conversations_state],
    #     outputs=[chatbot, graph_output, msg_input, all_conversations_state, toggle_graph_btn, showing_global_graph_state]
    # )

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
        print(f"Graph document directory '{graph_doc_dir}' will be allowed for serving files.")
        if GLOBAL_GRAPH_PATH: # Check if global graph path was successfully determined
             if not os.path.exists(GLOBAL_GRAPH_PATH):
                print(f"Warning: Global graph file '{GLOBAL_GRAPH_PATH}' does not exist. Please create it.")
        else:
            print(f"Warning: GLOBAL_GRAPH_PATH is not set, likely because GRAPH_DOCUMENT_DIRECTORY was not valid.")

    else:
        print("Warning: rag_system.GRAPH_DOCUMENT_DIRECTORY is not set. Graphs (including global) may not be served correctly unless their directories are manually added to allowed_paths.")

    # If GLOBAL_GRAPH_PATH is outside graph_doc_dir and needs its own directory to be allowed:
    # global_graph_dir = os.path.dirname(GLOBAL_GRAPH_PATH)
    # if global_graph_dir not in allowed_paths_list and os.path.isdir(global_graph_dir):
    # allowed_paths_list.append(global_graph_dir)
    allowed_paths_list.append("/data/users/pfont/final_documents")
    allowed_paths_list.append("/data/users/pfont/input")
    print(f"Launching Gradio app. Allowed paths for graphs: {allowed_paths_list}")
    demo.launch(allowed_paths=allowed_paths_list, share=False)