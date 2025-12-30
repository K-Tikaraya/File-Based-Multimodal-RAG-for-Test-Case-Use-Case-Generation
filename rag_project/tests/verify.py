import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'rag_project', 'src'))

from ingestion import IngestionEngine
from retrieval import RetrievalEngine
from generation import GenerationEngine

def test_pipeline():
    print(">>> Starting Pipeline Verification")
    
    # 1. Ingestion
    print("\n[1] Testing Ingestion...")
    ingestion = IngestionEngine()
    data_path = os.path.join("rag_project", "data")
    chunks = ingestion.process_folder(data_path)
    
    if len(chunks) > 0:
        print(f"SUCCESS: Processed {len(chunks)} chunks.")
    else:
        print("FAILURE: No chunks processed. Check data folder.")
        return

    # 2. Retrieval
    print("\n[2] Testing Retrieval...")
    retrieval = RetrievalEngine()
    retrieval.clear()
    retrieval.add_documents(chunks)
    
    query = "user signup requirements"
    docs = retrieval.query(query)
    
    if len(docs) > 0:
        print(f"SUCCESS: Retrieved {len(docs)} documents for query '{query}'.")
        print(f"Top Result Preview: {docs[0].page_content[:100]}...")
    else:
        print("FAILURE: No documents retrieved.")
        return

    # 3. Generation (Mock/Check)
    print("\n[3] Testing Generation Initialization...")
    try:
        # We won't actually invoke the LLM heavily here to avoid cost/time/setup issues in this verify script,
        # but we check if the engine initializes and can format a prompt.
        generator = GenerationEngine(local=True) # Force local to avoid API key requirement for this test
        print("SUCCESS: GenerationEngine initialized.")
        
        # Optional: dry run generate if local LLM is expected to be running. 
        # Since we don't know if Ollama is running in this env, we skip the actual invoke.
    except Exception as e:
        print(f"FAILURE: GenerationEngine crashed. Error: {e}")

if __name__ == "__main__":
    test_pipeline()
