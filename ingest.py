import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PDF_PATH = 'knowledge_base/sakshya_law.pdf'
PERSIST_DIR = 'data/index'

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'openai')  # 'openai', 'ollama', or 'gemini'

if EMBEDDING_MODEL == 'openai':
    embeddings = OpenAIEmbeddings()
elif EMBEDDING_MODEL == 'gemini':
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print(f"Loading PDF from {PDF_PATH}...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

print(f"Embedding and storing {len(docs)} chunks in ChromaDB...")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
vectorstore.persist()
print(f"Ingestion complete. Index stored at {PERSIST_DIR}") 