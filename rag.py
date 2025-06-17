import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
load_dotenv()

all_pdf=[]
pdf_dir="C:\\Users\\sonas\\OneDrive\\Documents\\GitHub\\digi-vakeel\\Laws"

for filename in os.listdir(pdf_dir):
        loader = PyPDFLoader( os.path.join(pdf_dir, filename))
        docs = loader.load()
        all_pdf.extend(docs)
        
# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(all_pdf)

# Create embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="C:\\Users\\sonas\\OneDrive\\Documents\\GitHub\\digi-vakeel\\vectorstore")

vectorstore.persist()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})