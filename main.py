from langchain_ollama import ChatOllama
import rag_implementation as rag
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableBranch
from langchain_huggingface import HuggingFaceImageToText

#Models
base_llm = ChatOllama(
    model="digi-vakeel",
    temperature=0
)

vision_model = HuggingFaceImageToText(
    model_id="microsoft/trocr-large-printed",
    model_kwargs={"max_length": 512}
)

#Image Handling
def process_image(input_dict: dict) -> str:
    image_object = input_dict.get("image_object")
    query = input_dict.get("query")

    if not image_object:
        return query
    transcribed_text = vision_model.invoke(image_object)

    formatted_query = f"""[IMAGE_TRANSCRIPTION]:
    {transcribed_text}

    [USER_QUERY]:
    {query}"""

    return formatted_query

#Text Handling with RAG
def process_text(query: str) -> str:
    qa_chain = RetrievalQA.from_chain_type(
       llm=base_llm,
       chain_type="stuff",
       retriever= rag.vector_store.as_retriever(search_kwargs={"k": 3}),
       return_source_documents=True,
    )
    return qa_chain.invoke(query)

base_chain = process_text
vision_chain = process_image | process_text


#result = qa_chain.invoke("i gave my google pixel 7a to a google authorized store but instead of reparing the phone the phone was furthur damaged what can i do legally?")
#rint(result['result'])

full_chain = RunnableBranch(
    (
        lambda x: "image_object" in x and x.get("image_object") is not None,
        vision_chain
    ),
    base_chain
)
      