import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from ingestion import IngestionEngine
from retrieval import RetrievalEngine
from generation import GenerationEngine
from utils import setup_logger

load_dotenv()
logger = setup_logger()

# --- CONSTANTS ---
DATA_FOLDER = "rag_data_source"  # Folder where uploaded files will be stored
DB_FOLDER = "chroma_db"          # Folder where vector database is stored

# Page Setup
st.set_page_config(page_title="RAG Test Case Generator", layout="wide")
st.title(" File-Based Multimodal RAG")

# Sidebar - Configuration
with st.sidebar:
    st.header("Configuration")
    
    # --- Model Selection (Same as before) ---
    model_choice = st.radio("LLM Provider", ["OpenAI (gpt-4o-mini)", "Groq (llama3-70b)", "Local (Ollama/Llama3)"])
    
    provider = "openai"
    api_key = None
    
    if "Groq" in model_choice:
        provider = "groq"
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
             api_key = st.text_input("Groq API Key", type="password")
    elif "Local" in model_choice:
        provider = "local"
    else:
        provider = "openai"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
             api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    
    # --- NEW: File Uploader Logic ---
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload context files (PDF, TXT, Images)", 
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "png", "jpg", "jpeg", "json", "yaml","docx","doc"]
    )
    
    if st.button("Ingest Files"):
        if not uploaded_files:
            st.warning("Please upload some files first.")
        else:
            with st.spinner("Processing files..."):
                try:
                    # 1. Reset/Create Data Folder
                    if os.path.exists(DATA_FOLDER):
                        shutil.rmtree(DATA_FOLDER)
                    os.makedirs(DATA_FOLDER)
                    
                    # 2. Save Uploaded Files to Disk
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    st.toast(f"Saved {len(uploaded_files)} files to disk.", icon="💾")

                    # 3. Initialize Engines
                    # (Ensure your IngestionEngine uses the robust code from our previous chat!)
                    ingestion = IngestionEngine() 
                    retrieval = RetrievalEngine(persist_directory=DB_FOLDER)
                    
                    # 4. Clear Old DB & Process New Files
                    retrieval.clear() 
                    chunks = ingestion.process_folder(DATA_FOLDER)
                    
                    # 5. Add to Vector DB
                    retrieval.add_documents(chunks)
                    
                    st.success(f"Ingestion Complete! Processed {len(chunks)} chunks.")
                    
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                    logger.error(f"Ingestion failed: {e}")

# Main Area - Query (Same as before)
st.header("Generate Test Cases")
query = st.text_area("Enter your requirement or query:", value="Create use cases for user signup", height=100)

if st.button("Generate"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Thinking..."):
            try:
                retrieval = RetrievalEngine(persist_directory=DB_FOLDER)
                # Retrieval
                docs = retrieval.query(query, top_k=5)
                context_texts = [d.page_content for d in docs]
                
                if not docs:
                    st.warning("No relevant context found in uploaded files.")
                
                # Generation
                generator = GenerationEngine(provider=provider, api_key=api_key)
                result = generator.generate(query, context_texts)
                
                # Render Results
                st.subheader("Generated Test Suite")
                if result.missing_information:
                    st.warning(f"Missing Info: {result.missing_information}")
                
                for i, uc in enumerate(result.use_cases):
                    with st.expander(f"Use Case {i+1}: {uc.title}", expanded=True):
                        st.markdown(f"**Goal**: {uc.goal}")
                        st.markdown(f"**Preconditions**: {', '.join(uc.preconditions)}")
                        st.markdown("**Steps**:")
                        for step in uc.steps:
                            st.write(f"- {step}")
                        st.markdown("**Expected Results**:")
                        for res in uc.expected_results:
                            st.write(f"- {res}")
                        if uc.negative_cases:
                            st.markdown("**Negative Cases**:")
                            for neg in uc.negative_cases:
                                st.write(f"- {neg}")

                # Debug View
                st.divider()
                with st.expander("🕵️ Debug: Retrieved Context"):
                    for i, doc in enumerate(docs):
                        st.caption(f"Source: {doc.metadata.get('source', 'unknown')}")
                        st.text(doc.page_content[:500] + "...") # Truncate for cleaner view
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")