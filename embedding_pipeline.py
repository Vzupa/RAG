import os
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
# Text Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# Vector Store (for saving/batching)
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

class EmbeddingPipeline:
    def __init__(self, use_openai: bool = True):
        """
        Initialize the pipeline.
        :param use_openai: If True, uses OpenAI Embeddings. If False, uses HuggingFace (local).
        """
        self.documents = []
        self.chunks = []
        
        # Initialize Embedding Model based on selection
        if use_openai:
            # Recommended: text-embedding-3-small is efficient and cheap
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
            print("--- Initialized OpenAI Embeddings (text-embedding-3-small) ---")
        else:
            # Alternative: all-MiniLM-L6-v2 is standard for efficient local CPU use
            self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("--- Initialized HuggingFace Embeddings (all-MiniLM-L6-v2) ---")

    def load_pdf(self, file_path: str):
        """Loads a PDF file."""
        print(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        self.documents.extend(docs)
        return docs

    def load_website(self, url: str):
        """Loads content from a URL."""
        print(f"Loading URL: {url}")
        loader = WebBaseLoader(url)
        docs = loader.load()
        self.documents.extend(docs)
        return docs

    def split_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Splits loaded documents into chunks.
        """
        print(f"Splitting documents (Chunk Size: {chunk_size}, Overlap: {chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""] # Tries to split by paragraph first
        )
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Created {len(self.chunks)} chunks.")
        return self.chunks

    def create_vector_db(self, persist_directory="./chroma_db"):
        """
        Generates embeddings in batch and stores them in ChromaDB.
        """
        print("Generating embeddings and storing in Vector DB...")
        
        # Chroma handles the batch embedding generation automatically
        vector_db = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embedding_model,
            persist_directory=persist_directory
        )
        print(f"Success! Vector DB created at {persist_directory}")
        return vector_db

# --- Example Usage (for testing) ---
if __name__ == "__main__":

    print("--- Starting Pipeline ---")
    # Ensure OPENAI_API_KEY is in your .env file
    pipeline = EmbeddingPipeline(use_openai=True)
    
    print("--- Loading Data ---")
    # 1. Load Data
    # Make sure this path is correct on your machine
    pipeline.load_pdf(r"Documents\El Husseini et al. - 2024 - Advanced Machine Learning Approaches for Zero-Day Attack Detection A Review.pdf") 
    
    # 2. Split Data
    pipeline.split_documents(chunk_size=1000, chunk_overlap=200)
    
    # 3. Embed and Store
    db = pipeline.create_vector_db()
    
    print("\n" + "="*40)
    print("Database Ready! Start asking questions.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*40 + "\n")

    # --- THE LOOP STARTS HERE ---
    while True:
        # 1. Get user input
        query = input("You: ")
        
        # 2. Check for exit condition
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting...")
            break
        
        # 3. Perform the search
        results = db.similarity_search(query)

        # Handle case where no results are found
        if not results:
            print("System: No relevant documents found.\n")
            continue

        top_result = results[0].page_content

        # 4. Print to terminal
        print(f"\nSystem (Top Result): \n{top_result}\n")
        print("-" * 20)

        # 5. Save to file (Append mode 'a')
        # We use 'a' so we don't overwrite previous answers
        with open("result.txt", "a", encoding="utf-8") as f:
            f.write(f"Question: {query}\n")
            f.write(f"Answer: {top_result}\n")
            f.write("-" * 40 + "\n\n")

        print("Saved to result.txt\n")
