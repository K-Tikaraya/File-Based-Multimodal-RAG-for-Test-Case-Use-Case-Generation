
import os
import sys
import pytest
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'rag_project', 'src'))
from generation import GenerationEngine

load_dotenv()

def test_groq_instantiate():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("SKIPPING: GROQ_API_KEY not found in env.")
        return

    print("Testing Groq Instantiation...")
    try:
        engine = GenerationEngine(provider="groq", api_key=api_key)
        print("SUCCESS: GenerationEngine instantiated with Groq.")
    except Exception as e:
        print(f"FAILURE: {e}")
        raise e

def test_groq_generation():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("SKIPPING: GROQ_API_KEY not found.")
        return

    print("Testing Groq Generation...")
    engine = GenerationEngine(provider="groq", api_key=api_key)
    
    query = "Test query"
    context = ["This is a test context."]
    
    try:
        result = engine.generate(query, context)
        print("SUCCESS: Generation completed.")
        print(f"Result Use Cases: {len(result.use_cases)}")
    except Exception as e:
        print(f"FAILURE: {e}")
        raise e

if __name__ == "__main__":
    test_groq_instantiate()
    # test_groq_generation() # Uncomment to run actual generation (costs tokens/requires valid key)
