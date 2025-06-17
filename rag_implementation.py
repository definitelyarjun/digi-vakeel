import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

pdf_directory = "C:/Users/arjun/OneDrive/Documents/digi-vakeel/laws"
all_pdf = []

for filename in os.listdir(pdf_directory):
    loader = PyPDFLoader(os.path.join(pdf_directory, filename))
    documents = loader.load()
    all_pdf.extend(documents)

#Split the documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = splitter.split_documents(all_pdf)

#Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

#Vector Store
vector_store = Chroma.from_documents(
    split_documents, 
    embeddings, 
    persist_directory="C:/Users/arjun/OneDrive/Documents/digi-vakeel/vector_store"
)
vector_store.persist()

retriever = vector_store.as_retriever(search_kwargs={"k": 5})