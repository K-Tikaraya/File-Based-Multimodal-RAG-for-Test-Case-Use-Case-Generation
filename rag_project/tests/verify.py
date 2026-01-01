import sys
import os
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.services.ingestion import IngestionEngine
from src.services.retrieval import RetrievalEngine
from src.services.generation import GenerationEngine

load_dotenv()

def test_pipeline():
    print(">>> Starting Pipeline Verification")
    
    # 1. Ingestion
    print("\n[1] Testing Ingestion...")
    try:
        ingestion = IngestionEngine()
        data_path = os.path.join("rag_data_source")
        if not os.path.exists(data_path):
             os.makedirs(data_path)
             # Create a dummy file if empty
             with open(os.path.join(data_path, "test.txt"), "w") as f:
                 f.write("This is a test context about user login. User needs email and password.")

        chunks = ingestion.process_folder(data_path)
        
        if len(chunks) > 0:
            print(f"SUCCESS: Processed {len(chunks)} chunks.")
        else:
            print("WARNING: No chunks processed. (Is the data folder empty?)")
            
    except Exception as e:
        print(f"FAILURE: Ingestion error: {e}")
        return

    # 2. Retrieval
    print("\n[2] Testing Retrieval...")
    try:
        retrieval = RetrievalEngine()
        retrieval.clear()
        retrieval.add_documents(chunks)
        
        query = "user login requirements"
        docs = retrieval.query(query)
        
        if len(docs) > 0:
            print(f"SUCCESS: Retrieved {len(docs)} documents for query '{query}'.")
            print(f"Top Result Preview: {docs[0].page_content[:100]}...")
        else:
            print("WARNING: No documents retrieved (might be due to empty folder or high threshold).")
            
    except Exception as e:
        print(f"FAILURE: Retrieval error: {e}")
        return

    # 3. Generation
    print("\n[3] Testing Generation Initialization...")
    try:
        # We try to init. If API key missing, it raises ValueError
        if not os.getenv("GROQ_API_KEY"):
            print("WARNING: GROQ_API_KEY missing. Skipping actual Generation check.")
            return

        generator = GenerationEngine() # defaults to groq
        print("SUCCESS: GenerationEngine initialized.")
        
        # Optional: dry run?
        # result = generator.generate("login tests", ["context..."])
        # print("Generation output:", result)
        
    except Exception as e:
        print(f"FAILURE: GenerationEngine crashed. Error: {e}")

if __name__ == "__main__":
    test_pipeline()
