from langchain_ollama import ChatOllama
import rag_implementation as rag
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnableMap
from operator import itemgetter
from transformers import pipeline
from PIL import Image
import langdetect
from sarvamai import SarvamAI
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# memory imports
from memory_history import BufferWindowMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate

load_dotenv()

# Chat map to store histories
chat_map = {}
def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
    print(f"Getting history for session_id={session_id} with k={k}")
    print(chat_map)
    if session_id not in chat_map:
        chat_map[session_id] = BufferWindowMessageHistory(k=k)
    return chat_map[session_id]
api_key = os.getenv("SARVAM_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")

# Models
# base_llm = ChatOllama(
#     model="gemma3:1b",
#     temperature=0
# )

base_llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

ocr_model = pipeline("image-to-text", model="microsoft/trocr-large-printed")

client = SarvamAI(api_subscription_key=api_key)

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

# Language and image utilities
def process_image(input_dict: dict) -> str:
    image_object = input_dict.get("image_object")
    query = input_dict.get("query")
    if not isinstance(image_object, Image.Image):
        return query
    transcribed_text = ocr_model(image_object)[0]['generated_text']
    return f"""[IMAGE_TRANSCRIPTION]:\n{transcribed_text}\n\n[USER_QUERY]:\n{query}"""

def is_english(query: str) -> bool:
    try:
        return langdetect.detect(query) == 'en'
    except:
        return True

def translate_to_ml(text: str) -> str:
    response = client.text.translate(
        input=text,
        source_language_code="en-IN",
        target_language_code="ml-IN",
        model="sarvam-translate:v1",
    )
    return response.translated_text

def translate_to_en(text: str) -> str:
    response = client.text.translate(
        input=text,
        source_language_code="ml-IN",
        target_language_code="en-IN",
        model="sarvam-translate:v1",
    )
    return response.translated_text

# Branch chains
vision_chain = RunnableLambda(process_image) | rag_chain

vision_malayalam_chain = (
    RunnableLambda(lambda x: f"[IMAGE_TRANSCRIPTION]:\n{ocr_model(x['image_object'])[0]['generated_text']}\n\n[USER_QUERY]:\n{translate_to_en(x['query'])}")
    | rag_chain
    | itemgetter("result")
    | RunnableLambda(translate_to_ml)
    | (lambda text: {"result": text})
)

base_malayalam_chain = (
    RunnableLambda(lambda x: {"query": translate_to_en(x["query"]), "history": x.get("history", [])})
    | rag_chain
    | itemgetter("result")
    | RunnableLambda(translate_to_ml)
    | (lambda text: {"result": text})
)

# base_chain = itemgetter("query") | rag_chain
base_chain = RunnableLambda(lambda x: x) | rag_chain

full_chain = RunnableBranch(
    (lambda x: "image_object" in x and x.get("image_object") is not None and not is_english(x.get("query", "")), vision_malayalam_chain),
    (lambda x: "image_object" in x and x.get("image_object") is not None, vision_chain),
    (lambda x: not is_english(x.get("query", "")), base_malayalam_chain),
    base_chain
)

# Memory wrapper
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
