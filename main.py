from langchain_ollama import ChatOllama
import rag_implementation as rag
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnableMap
from operator import itemgetter
from transformers import pipeline
from PIL import Image
import langdetect
from sarvamai import SarvamAI
from dotenv import load_dotenv
import os
from memory_history import BufferWindowMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate

chat_map = {}
def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
    print(f"Getting history for session_id={session_id} with k={k}")
    print(chat_map)
    if session_id not in chat_map:
        chat_map[session_id] = BufferWindowMessageHistory(k=k)
    return chat_map[session_id]

load_dotenv()
sarvam_api_key = os.getenv("SARVAM_API_KEY")

#Models
base_llm = ChatOllama(
    model="digi-vakeel",
    temperature=0
)

ocr_model = pipeline("image-to-text", model="microsoft/trocr-large-printed")

client = SarvamAI(api_subscription_key=sarvam_api_key)

retriever = rag.vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5, "fetch_k": 10}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chat Prompt with RAG context and history
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a legal assistant specialized in Indian consumer laws. Use the following context to answer user questions.\n\n{context}"
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{query}")
])

# Core RAG+Memory chat chain
rag_chain = (
    RunnableMap({
        "context": lambda x: format_docs(retriever.get_relevant_documents(x["query"])),
        "query": lambda x: x["query"],
        "history": lambda x: x.get("history", [])
    })
    | chat_prompt
    | base_llm
    | RunnableLambda(lambda x: {"result": x.content})
)

#Image Handling
def process_image(input_dict: dict) -> str:
    image_object = input_dict.get("image_object")
    query = input_dict.get("query")

    if not isinstance(image_object, Image.Image):
        return query

    transcribed_text = ocr_model(image_object)[0]['generated_text']

    formatted_query = f"""[IMAGE_TRANSCRIPTION]:
    {transcribed_text}

    [USER_QUERY]:
    {query}"""

    return formatted_query

#Language check
def is_english(query: str) -> str:
    if langdetect.detect(query) == 'en':
        return True
    return False

#Translation (Only malayalam for now)
def translate_to_ml(text: str) -> str:
    response = client.text.translate(
      input= text,
      source_language_code = "en-IN",
      target_language_code="ml-IN",
      model="sarvam-translate:v1",
    )
    return response.translated_text

def translate_to_en(text: str) -> str:
    response = client.text.translate(
      input = text,
      source_language_code="ml-IN",
      target_language_code="en-IN",
      model="sarvam-translate:v1",
    )
    return response.translated_text

#Chains
vision_malayalam_chain = (
    RunnableLambda(lambda x: f"[IMAGE_TRANSCRIPTION]:\n{ocr_model(x['image_object'])[0]['generated_text']}\n\n[USER_QUERY]:\n{translate_to_en(x['query'])}")
    | rag_chain
    | itemgetter("result")
    | RunnableLambda(translate_to_ml)
    | (lambda text: {"result": text})
)
vision_chain = RunnableLambda(process_image) | rag_chain

base_malayalam_chain = (
    RunnableLambda(lambda x: {"query": translate_to_en(x["query"]), "history": x.get("history", [])})
    | rag_chain
    | itemgetter("result")
    | RunnableLambda(translate_to_ml)
    | (lambda text: {"result": text})
)
base_chain = RunnableLambda(lambda x: x) | rag_chain

#Branch
full_chain = RunnableBranch(
    (lambda x: "image_object" in x and x.get("image_object") is not None and not is_english(x.get("query", "")), vision_malayalam_chain),
    (lambda x: "image_object" in x and x.get("image_object") is not None, vision_chain),
    (lambda x: not is_english(x.get("query", "")), base_malayalam_chain),
    base_chain
)

full_chain_with_memory = RunnableWithMessageHistory(
    full_chain,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="default_session",
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in memory",
            default=4,
        )
    ]
)

__all__ = ["full_chain_with_memory"]