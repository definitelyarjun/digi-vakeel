import gradio as gr
from PIL import Image
from main import full_chain

print("Gradio App is ready to launch.")

def run_rag_pipeline(query: str, image: Image.Image | None):
 
    if not query:
        return "Please provide a query."

    if image:
        input_dict = {"query": query, "image_object": image}
    else:
        input_dict = {"query": query}

    result = full_chain.invoke(input_dict)
    return result.get("result", "Sorry, I could not find an answer.")

interface = gr.Interface(
    fn=run_rag_pipeline,
    inputs=[
        gr.Textbox(lines=3, label="Query", placeholder="Enter your question here..."),
        gr.Image(type="pil", label="Upload an Image (Optional)")
    ],
    outputs=gr.Textbox(label="Answer", lines=8),
    title="Modular RAG Pipeline",
    description="Ask a question about your documents. Optionally, upload an image (like a receipt or screenshot) to have its text included in the query.",
    allow_flagging="never",
    examples=[
        ["I gave my google pixel to a store for repair, but they damaged it further. What can I do?", None],
        ["What are my rights regarding this service request?", "sample_receipt.png"], # You can pre-load examples
    ]
)

interface.launch()