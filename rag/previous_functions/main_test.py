import gradio as gr

def chat_search(message, history):
    response = f"Echo: {message}"
    history = history + [(message, response)]
    
    # Generate HTML graph
    html_path = "/data/users/pfont/graph/online_knowledge_graph.html"
    with open(html_path, "r") as f:
        html_content = f.read()

    return history, html_content

_, html = chat_search("q",[])

with gr.Blocks() as demo:
    gr.Markdown("# Chatbot Histórico \nPregunta sobre documentos históricos. Las preguntas deben limitarse a consultas sobre personas o eventos especificos. Se mostrará todo el contexto usado para la respuesta.")

    state = gr.State([])
    # Para evitar que haya problemas si se borra el texto, antes de leerlo
    temp_user_message_holder = gr.Textbox(visible=False, label="temp_user_msg")

    with gr.Row():
        with gr.Column(scale=2):  
            chatbot = gr.Chatbot()
            input_text = gr.Textbox(placeholder="Type a message", show_label=False)

        with gr.Column(scale=1):  
            html_output = gr.HTML(value=html, label="Grafo de Conocimiento Dinámico", min_height=850)

    def handle_input(message, history):
        return chat_search(message, history)

    input_text.submit(handle_input, [input_text, state], [chatbot, html_output])
    input_text.submit(lambda x, y: "", None, input_text)  # clears textbox
    input_text.submit(lambda msg, hist: hist + [(msg, f"Echo: {msg}")], [input_text, state], state)

demo.launch()
