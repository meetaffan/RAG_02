"""
Worker module for processing RAG queries asynchronously
Uses RAG-01 functionality directly
"""
import sys
import os

# Add RAG-01 to Python path to import its modules
rag01_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'RAG-01')
sys.path.insert(0, rag01_path)

# Import RAG-01's main module functionality
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Configuration
pdf_path = os.getenv('PDF_PATH', os.path.join(rag01_path, 'nodejs.pdf'))
qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
qdrant_port = os.getenv('QDRANT_PORT', '6333')
qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

# Global RAG components (initialized once at module load)
embedding_model = None
vector_store = None
gemini_client = None

def initialize_rag(force: bool = False):
    """
    Initialize RAG system using Gemini embeddings API
    No local ML model = no fork issues with RQ workers
    """
    global embedding_model, vector_store, gemini_client
    
    # Vector Embeddings using Gemini API (replaces HuggingFace SentenceTransformer)
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    
    # Connect to existing vector store (don't reload PDF each time)
    client = QdrantClient(url=qdrant_url)
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(c.name == "rag_01" for c in collections)
    
    if not collection_exists:
        # First time: Load PDF and create collection
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents=docs)
        
        # Create vector store
        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            url=qdrant_url,
            collection_name="rag_01",
            embedding=embedding_model
        )
        print(f"✓ RAG initialized: {len(split_docs)} chunks indexed from {pdf_path}")
    else:
        # Collection exists: just connect to it
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="rag_01",
            embedding=embedding_model
        )
        print(f"✓ Connected to existing vector store: rag_01")
    
    # Initialize Gemini client
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# NOTE: initialize_rag() is called lazily on first query, not at module load.
# This prevents the app container (which also imports this module) from
# making unnecessary API calls to Gemini on startup.


def process_query(query: str):
    """
    Process a user query through the RAG system
    This function runs asynchronously in the RQ worker
    """
    global embedding_model, vector_store, gemini_client
    print(f"Processing query: {query}")

    # Lazy initialization — only runs in the worker process on first query
    if not vector_store or not gemini_client:
        try:
            initialize_rag()
        except Exception as e:
            print(f"✗ Error initializing RAG: {e}")
            return {"error": f"RAG init failed: {e}", "query": query}
    
    try:
        # Retrieve relevant documents (same as RAG-01)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        
        # Build context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}
Answer:"""
        
        # Get response from Gemini
        interaction = gemini_client.interactions.create(
            model="gemini-3-flash-preview",
            input=prompt
        )
        
        answer = interaction.outputs[-1].text
        
        # Extract sources
        sources = []
        for idx, doc in enumerate(relevant_docs, 1):
            page_no = doc.metadata.get('page', 'N/A')
            sources.append({
                "chunk": idx,
                "page": page_no
            })
        
        print(f"✓ Query processed successfully")
        return {
            "answer": answer,
            "sources": sources,
            "question": query
        }
    except Exception as e:
        print(f"✗ Error processing query: {e}")
        return {
            "error": str(e),
            "query": query
        }

    