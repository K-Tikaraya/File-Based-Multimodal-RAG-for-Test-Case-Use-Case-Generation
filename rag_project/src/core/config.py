import os
from dotenv import load_dotenv

load_dotenv()

# --- Model Constants ---

# Vision Models
VISION_MODELS = {
    "Maverick (High Detail/MoE)": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "Scout (Fast/High Throughput)": "meta-llama/llama-4-scout-17b-16e-instruct",
    "GPT-OSS (Creative Description)": "openai/gpt-oss-120b"
}

# Generation Models
GENERATION_MODELS = {
    "Llama 3.3 70B (Versatile)": "llama-3.3-70b-versatile",
    "GPT-OSS 120B": "openai/gpt-oss-120b"
}

# Guardrail Models
GUARDRAIL_MODELS = {
    "Llama Guard 4 (12B)": "meta-llama/llama-guard-4-12b",
    "Llama Prompt Guard 2 (22M)": "meta-llama/llama-prompt-guard-2-22m",
    "None": None
}

# Embedding Model (Local)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Paths ---
DATA_FOLDER = "rag_data_source"
DB_FOLDER = "chroma_db"

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
