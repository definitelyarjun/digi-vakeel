# proto_gradio.py (Backwards-compatible version)

import gradio as gr
from PIL import Image
import uuid
from main import full_chain_with_memory

print("Gradio App is ready to launch.")


def chat_with_rag(query: str, history: list, image_object: Image.Image | None, session_id: str | None):
    if session_id is None:
        session_id = str(uuid.uuid4())
        print(f"Starting new chat session: {session_id}")

    if not query and not image_object:
        return "Please provide a query or an image.", session_id

    input_dict = {"query": query}
    if image_object:
        input_dict["image_object"] = image_object
    
    config = {"configurable": {"session_id": session_id}}

    result = full_chain_with_memory.invoke(input_dict, config=config)
    response = result.get("result", "Sorry, I could not find an answer.")
    
    return response, session_id

def create_chat_interface():
    session_id_state = gr.State(None)

    def chat_wrapper(query, history, image_object):
        response, session_id_val = chat_with_rag(query, history, image_object, session_id_state.value)
        session_id_state.value = session_id_val
        return response

    interface = gr.ChatInterface(
        fn=chat_wrapper,
        additional_inputs=[
            gr.Image(type="pil", label="Upload an Image (Optional)")
        ],
        chatbot=gr.Chatbot(height=500, type='messages', bubble_full_width=False), # Fixed warning
        textbox=gr.Textbox(placeholder="Ask a question about Indian consumer law...", container=False, scale=7),
        title="Legal Assistant for Indian Consumer Law",
        description="Ask a question about your documents. Optionally, upload an image (like a receipt or screenshot) to have its text included in the query. Your conversation is remembered.",
    )
    return interface

if __name__ == "__main__":
    app = create_chat_interface()
    app.launch()