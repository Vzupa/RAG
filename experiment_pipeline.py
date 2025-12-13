import os
import shutil
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# --- CONFIGURATION MATRIX ---
# 1. Models to test
MODELS_TO_TEST = ["openai", "openai-large", "minilm", "bge"] 
CHUNK_SIZES = [500, 1000, 2000]
OVERLAPS = [50, 100, 200]
TEST_QUERIES = [
    'In the context of detecting encrypted malicious traffic, does converting packet flows into the frequency domain (via spectral analysis) retain sufficient resolution to distinguish "low-and-slow" exfiltration attacks from legitimate background "keep-alive" signals, or is this feature engineering approach statistically biased toward high-volume volumetric attacks?',
    'When applying Semi-Supervised Learning to zero-day intrusion detection, how does the "Cluster Assumption"—which posits that points in the same high-density region share the same label—mitigate or exacerbate the risk of "error propagation" when an unknown attack vector mimics the statistical distribution of benign traffic in the unlabeled dataset?',
    'Is the integration of FPGA-based hardware acceleration merely an optimization for reducing latency in network intrusion detection systems, or is it a functional prerequisite for deploying complex Deep Learning architectures (like Transformers) in an inline Prevention (IPS) mode at speeds exceeding 10Gbps without forcing packet bypass?',
    'Why might a "Living-off-the-Land" insider threat—where a privileged user employs standard administrative tools for malicious purposes—result in a False Negative when using statistical outlier detection methods, yet trigger a True Positive when utilizing User and Entity Behavior Analytics (UEBA) based on historical deviations?',
    'How does the "Cold Start" problem in behavioral analytics (where new users or devices lack a historical baseline) create a specific vulnerability window for zero-day attacks, and can this be effectively mitigated by using semi-supervised peer-group clustering as a temporary proxy for individual baselines?'
]

PDF_PATH = r"Documents"

class ExperimentPipeline:
    def __init__(self, model_type: str):
        self.documents = []
        self.chunks = []
        self.model_type = model_type
        
        if model_type == "openai":
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
            print("--- Initialized: OpenAI (text-embedding-3-small) ---")
            
        elif model_type == "openai-large":
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
            print("--- Initialized: OpenAI (text-embedding-3-large) ---")
            
        elif model_type == "minilm":
            self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("--- Initialized: HuggingFace (all-MiniLM-L6-v2) ---")
            
        elif model_type == "bge":
            self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
            print("--- Initialized: HuggingFace (BAAI/bge-small-en) ---")

    def load_data(self, folder_path):
            """Loads all PDF files from a specified folder."""
            if not os.path.exists(folder_path):
                print(f"Error: Folder '{folder_path}' not found.")
                return

            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            
            if not files:
                print(f"No PDFs found in {folder_path}")
                return

            print(f"Found {len(files)} PDFs in folder. Loading...")

            for filename in files:
                file_path = os.path.join(folder_path, filename)
                print(f" - Loading: {filename}")
                
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    self.documents.extend(docs)
                except Exception as e:
                    print(f"   Error loading {filename}: {e}")

    def split_and_embed(self, chunk_size, chunk_overlap, run_id):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.chunks = splitter.split_documents(self.documents)
        
        persist_dir = f"./experiments/db_{run_id}"
        
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            
        vector_db = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embedding_model,
            persist_directory=persist_dir
        )
        return vector_db

def run_experiments():
    results_file = "experiment_results.txt"
    
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("--- RAG EXPERIMENT REPORT ---\n\n")

    print(f"Starting Experiments... Output will be saved to {results_file}")

    for model in MODELS_TO_TEST:
        for size in CHUNK_SIZES:
            for overlap in OVERLAPS:
                
                run_id = f"{model}_{size}_{overlap}"
                print(f"Running Experiment: {run_id}...")

                pipeline = ExperimentPipeline(model_type=model)
                pipeline.load_data(PDF_PATH)
                
                db = pipeline.split_and_embed(size, overlap, run_id)

                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(f"=== EXPERIMENT: {run_id.upper()} ===\n")
                    f.write(f"Model: {model} | Chunk: {size} | Overlap: {overlap}\n")
                    f.write("-" * 40 + "\n")

                    for q in TEST_QUERIES:
                        results = db.similarity_search(q, k=3) 
                        
                        f.write(f"QUERY: {q}\n")
                        f.write("-" * 20 + "\n")
                        
                        for i, doc in enumerate(results):
                            source = doc.metadata.get('source', 'Unknown Source')
                            page = doc.metadata.get('page', 'Unknown Page')
                            
                            f.write(f"Result {i+1} (Source: {source}, Page: {page}):\n")
                            f.write(f"{doc.page_content}\n")
                            f.write("\n")
                        
                        f.write("\n")
                    
                    f.write("\n" + "="*50 + "\n\n")
                
                del db
                print(f"Finished {run_id}")

    print("All experiments done! Check experiment_results.txt")

if __name__ == "__main__":
    run_experiments()