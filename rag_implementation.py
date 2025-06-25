import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

pdf_directory = "./datasets/rag_pdf"
vector_store_directory = "./vector_store"
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

def create_or_get_vector_store():
    if os.path.exists(pdf_directory) and os.listdir(vector_store_directory):
        vector_store = Chroma(persist_directory=vector_store_directory, embedding_function=embeddings_model)
    else:
        all_pdf = []
        for filename in os.listdir(pdf_directory):
          loader = PyPDFLoader(os.path.join(pdf_directory, filename))
          documents = loader.load()
          all_pdf.extend(documents)

        #Split the documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = splitter.split_documents(all_pdf)

        #Embeddings
        embeddings = embeddings_model

        #Vector Store
        vector_store = Chroma.from_documents(
          split_documents, 
          embeddings, 
          persist_directory=vector_store_directory
        )
    return vector_store


"""
"""
vector_store = create_or_get_vector_store()

query = "what is section 107 of consumer protection laws 2019"
results_mmr = vector_store.max_marginal_relevance_search(query, k=3, fetch_k=10, lambda_mult=0.5)
for i, doc in enumerate(results_mmr):
  print(f"Result {i+1}:\n{doc.page_content}\n")