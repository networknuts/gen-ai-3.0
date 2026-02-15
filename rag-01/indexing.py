from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# ENVIRONMENT SETUP

load_dotenv()

# CONFIGURATION

PDF_FILE = "data.pdf"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ansible_vectors"
EMBEDDING_MODEL = "text-embedding-3-large"

# STEP 1: LOAD THE PDF DOCUMENT

loader = PyPDFLoader(file_path=PDF_FILE)
documents = loader.load()

# STEP 2: BREAK PDF INTO CHUNKS

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)

chunked_documents = text_splitter.split_documents(documents) #split_text did not work, we used split_documents

# STEP 3: CHOOSE MY EMBEDDING MODEL

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large")

# STEP 4: STORE CHUNKS IN VECTOR DB

qdrant = QdrantVectorStore.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME
)

print("Documents indexed and stored successfully!")
