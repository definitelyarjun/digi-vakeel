from langchain_ollama import ChatOllama
import rag_implementation as rag
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from operator import itemgetter
from transformers import pipeline
from PIL import Image
import langdetect
from sarvamai import SarvamAI
from dotenv import load_dotenv
from google.cloud import translate
import os

load_dotenv()
api_key = os.getenv("SARVAM_API_KEY")

#Models
base_llm = ChatOllama(
    model="digi-vakeel",
    temperature=0
)

ocr_model = pipeline("image-to-text", model="microsoft/trocr-large-printed")

client = SarvamAI(api_subscription_key=api_key)

qa_chain = RetrievalQA.from_chain_type(
   llm=base_llm,
   chain_type="stuff",
   retriever= rag.vector_store.as_retriever(search_type="mmr",search_kwargs={"k": 3, "lambda_mult": 0.5, "fetch_k": 10}),
   return_source_documents=True,
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
def is_english(query: str) -> bool:
    if langdetect.detect(query) == 'en':
        return True
    return False

#Translation (Only malayalam for now)
def translate_to_ml(model_response: str) -> str:
    response = client.text.translate(
      input= model_response,
      source_language_code = "en-IN",
      target_language_code="ml-IN",
      model="sarvam-translate:v1",
    )
    return response.translated_text

def translate_to_en(query: str) -> str:
    response = client.text.translate(
      input = query,
      source_language_code="ml-IN",
      target_language_code="en-IN",
      model="sarvam-translate:v1",
    )
    return response.translated_text

#Chains
vision_malayalam_chain = (
    RunnableLambda(
        lambda x: f"[IMAGE_TRANSCRIPTION]:\n{ocr_model(x['image_object'])[0]['generated_text']}\n\n[USER_QUERY]:\n{translate_to_en(x['query'])}"
    )
    | qa_chain
    | itemgetter("result")
    | RunnableLambda(translate_to_ml)
    | (lambda text: {"result": text})
)
vision_chain = RunnableLambda(process_image) | qa_chain
base_malayalam_chain = (
    itemgetter("query")
    | RunnableLambda(translate_to_en)
    | qa_chain
    | itemgetter("result")
    | RunnableLambda(translate_to_ml)
    | (lambda text: {"result": text})
)
base_chain = itemgetter("query") | qa_chain

#Branch
full_chain = RunnableBranch(
    (
        
        lambda x: "image_object" in x and x.get("image_object") is not None and not is_english(x.get("query", "")), #Image + Malayalam
        vision_malayalam_chain
    ),
    (
        lambda x: "image_object" in x and x.get("image_object") is not None, #Image + English
        vision_chain
    ),
    (
        lambda x: not is_english(x.get("query", "")), #Malayalam
        base_malayalam_chain
    ),
    base_chain #English
)
      