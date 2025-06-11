import gradio as gr
from datetime import datetime
import os
import uuid
import time
import json
from main import RAGSystem

# Global state for chat history and graph path
chat_history_global = []
graph_path_global = None
rag_system = RAGSystem()
rag_system.initialize_models_and_data()

GLOBAL_GRAPH_FILENAME = "global_knowledge_graph.html"
GLOBAL_GRAPH_PATH = None
CONVERSATION_SAVE_DIR = "saved_conversations"

if rag_system.GRAPH_DOCUMENT_DIRECTORY and os.path.isdir(rag_system.GRAPH_DOCUMENT_DIRECTORY):
    GLOBAL_GRAPH_PATH = os.path.join(rag_system.GRAPH_DOCUMENT_DIRECTORY, GLOBAL_GRAPH_FILENAME)
else:
    print("Warning: GRAPH_DOCUMENT_DIRECTORY not valid. Global graph feature may not work.")

os.makedirs(CONVERSATION_SAVE_DIR, exist_ok=True)

def create_new_conversation_entry(name):
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "history": [],
        "graph_path": None
    }

def get_conversation_by_id(conv_id, conversations_list):
    for conv in conversations_list:
        if conv["id"] == conv_id:
            return conv
    return None

def save_conversation_to_disk(conversation):
    filename = os.path.join(CONVERSATION_SAVE_DIR, f"{conversation['id']}.json")
    with open(filename, 'w') as f:
        json.dump(conversation, f, indent=2)
    return filename

def load_all_saved_conversations():
    conversations = []
    for file in os.listdir(CONVERSATION_SAVE_DIR):
        if file.endswith(".json"):
            try:
                with open(os.path.join(CONVERSATION_SAVE_DIR, file), 'r') as f:
                    conv = json.load(f)
                    conversations.append(conv)
            except Exception as e:
                print(f"Error loading conversation {file}: {e}")
    return sorted(conversations, key=lambda x: x.get("name", ""))

def send_message(user_input, chat_history):
    if not user_input.strip():
        yield chat_history, "", ""
        return

    updated_history = chat_history + [[user_input, None]]
    yield updated_history, "", ""

    try:
        bot_response, new_graph_path = rag_system.chat_search(user_input, chat_history)
    except Exception as e:
        bot_response = f"Error al procesar la consulta: {str(e)}"
        new_graph_path = None

    updated_history[-1][1] = bot_response
    yield updated_history, get_graph_html(new_graph_path), ""

def get_graph_html(graph_file_path):
    if not graph_file_path or not os.path.exists(graph_file_path):
        return "<p>No hay grafo disponible.</p>"
    abs_path = os.path.abspath(graph_file_path)
    try:
        iframe_src = f"/gradio_api/file={abs_path}?v={time.time()}"
        return f'<iframe src="{iframe_src}" width="100%" height="600px" style="border:none;"></iframe>'
    except Exception as e:
        return f"<p>Error al cargar el grafo: {str(e)}</p>"

def rename_conversation(selected_conv_id, new_name, all_conversations):
    updated = [conv.copy() for conv in all_conversations]
    for conv in updated:
        if conv["id"] == selected_conv_id:
            conv["name"] = new_name
            break
    return updated, [(c["name"], c["id"]) for c in updated]

def handle_new_conversation_click(current_conversations):
    new_conv_name = f"Conversación {len(current_conversations) + 1}"
    new_conv = create_new_conversation_entry(new_conv_name)
    updated_conversations = current_conversations + [new_conv]
    choices = [(c["name"], c["id"]) for c in updated_conversations]
    return (
        [], chatbot, get_graph_html(None),
        gr.update(choices=choices, value=new_conv["id"]),
        updated_conversations, new_conv["id"],
        gr.update(value="Ver Grafo Global"), False
    )

def handle_switch_conversation_change(selected_conv_id, current_conversations):
    active_conv = get_conversation_by_id(selected_conv_id, current_conversations)
    if active_conv:
        return (
            active_conv["history"],
            get_graph_html(active_conv["graph_path"]),
            active_conv["id"],
            gr.update(value="Ver Grafo Global"), False
        )
    return [], get_graph_html(None), selected_conv_id, gr.update(), False

def handle_toggle_graph_view(is_currently_showing_global, active_conv_id, all_conversations):
    new_showing_global = not is_currently_showing_global
    button_text = "Ver Grafo Actual" if new_showing_global else "Ver Grafo Global"
    graph_html_content = ""
    if new_showing_global:
        graph_html_content = get_graph_html(GLOBAL_GRAPH_PATH)
    else:
        active_conv = get_conversation_by_id(active_conv_id, all_conversations)
        if active_conv:
            graph_html_content = get_graph_html(active_conv["graph_path"])
    return graph_html_content, gr.update(value=button_text), new_showing_global

def on_send_message_submit(user_input, active_conv_id, current_conversations_list_state):
    active_conv_index = next((i for i, c in enumerate(current_conversations_list_state) if c["id"] == active_conv_id), -1)
    if active_conv_index == -1:
        yield current_conversations_list_state[0]["history"], get_graph_html(None), "", current_conversations_list_state, gr.update(), False
        return

    updated_conversations = [conv.copy() for conv in current_conversations_list_state]
    active_conv = updated_conversations[active_conv_index]

    active_conv["history"] = active_conv["history"] + [[user_input, None]]
    yield active_conv["history"], get_graph_html(active_conv["graph_path"]), "", updated_conversations, gr.update(), False

    try:
        bot_response, new_graph_path = rag_system.chat_search(user_input, active_conv["history"])
    except Exception as e:
        bot_response = f"Error al procesar la consulta: {str(e)}"
        new_graph_path = active_conv["graph_path"]

    active_conv["history"][-1][1] = bot_response
    active_conv["graph_path"] = new_graph_path
    yield active_conv["history"], get_graph_html(active_conv["graph_path"]), "", updated_conversations, gr.update(), False

with gr.Blocks(title="Chatbot Histórico con Grafo", theme="citrus") as demo:
    initial_conversation = create_new_conversation_entry("Conversación 1")
    all_conversations_state = gr.State([initial_conversation])
    active_conversation_id_state = gr.State(initial_conversation["id"])
    showing_global_graph_state = gr.State(False)

    gr.Markdown("## Chatbot Histórico con Visualización de Conocimiento\nPregunta sobre documentos históricos.")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### Conversaciones")
            conversation_selector = gr.Radio(
                label="Selecciona una conversación",
                choices=[(initial_conversation["name"], initial_conversation["id"])],
                value=initial_conversation["id"],
                type="value"
            )
            new_conv_btn = gr.Button("➕ Nueva Conversación")
            rename_conv_input = gr.Textbox(label="Nuevo nombre para la conversación")
            rename_conv_btn = gr.Button("Cambiar Nombre")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                bubble_full_width=False,
                height=550,
                render="dynamic",
                autoscroll=True
            )
            msg_input = gr.Textbox(
                placeholder="Escribe tu pregunta aquí...",
                label="Tu Pregunta",
                show_label=False,
                lines=2
            )
            send_btn = gr.Button("Enviar")
            typing_indicator = gr.HTML("<div id='typing-indicator' style='display: none;'>Bot está escribiendo...</div>")

        with gr.Column(scale=2, min_width=300):
            with gr.Row():
                gr.Markdown("### Grafo de Conocimiento")
                toggle_graph_btn = gr.Button("Ver Grafo Global", scale=1)
            graph_output = gr.HTML()

    def load_initial_ui(active_id, conversations):
        conv = get_conversation_by_id(active_id, conversations)
        initial_history = []
        initial_graph_html = get_graph_html(None)
        if conv:
            initial_history = conv["history"]
            initial_graph_html = get_graph_html(conv["graph_path"])
        return initial_history, initial_graph_html, "Ver Grafo Global", False

    demo.load(fn=load_initial_ui, inputs=[active_conversation_id_state, all_conversations_state], outputs=[chatbot, graph_output, toggle_graph_btn, showing_global_graph_state])

    new_conv_btn.click(fn=handle_new_conversation_click, inputs=[all_conversations_state], outputs=[
        chatbot, graph_output, msg_input,
        conversation_selector, all_conversations_state, active_conversation_id_state,
        toggle_graph_btn, showing_global_graph_state
    ])

    rename_conv_btn.click(fn=rename_conversation, inputs=[active_conversation_id_state, rename_conv_input, all_conversations_state], outputs=[
        all_conversations_state, conversation_selector
    ])

    conversation_selector.change(fn=handle_switch_conversation_change, inputs=[conversation_selector, all_conversations_state], outputs=[
        chatbot, graph_output, active_conversation_id_state, toggle_graph_btn, showing_global_graph_state
    ])

    msg_input.submit(fn=on_send_message_submit, inputs=[msg_input, active_conversation_id_state, all_conversations_state], outputs=[
        chatbot, graph_output, msg_input, all_conversations_state, toggle_graph_btn, showing_global_graph_state
    ])

    send_btn.click(fn=on_send_message_submit, inputs=[msg_input, active_conversation_id_state, all_conversations_state], outputs=[
        chatbot, graph_output, msg_input, all_conversations_state, toggle_graph_btn, showing_global_graph_state
    ])

    toggle_graph_btn.click(fn=handle_toggle_graph_view, inputs=[showing_global_graph_state, active_conversation_id_state, all_conversations_state], outputs=[
        graph_output, toggle_graph_btn, showing_global_graph_state
    ])

if __name__ == "__main__":
    allowed_paths_list = []
    if rag_system.GRAPH_DOCUMENT_DIRECTORY:
        allowed_paths_list.append(rag_system.GRAPH_DOCUMENT_DIRECTORY)
    demo.launch(allowed_paths=allowed_paths_list)