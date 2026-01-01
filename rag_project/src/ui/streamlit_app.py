import streamlit as st
import os
import shutil
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import (
    VISION_MODELS, 
    GENERATION_MODELS, 
    GUARDRAIL_MODELS, 
    DATA_FOLDER, 
    DB_FOLDER,
    GROQ_API_KEY
)
from services.ingestion import IngestionEngine
from services.retrieval import RetrievalEngine
from services.generation import GenerationEngine
from core.logger import setup_logger

logger = setup_logger()

st.set_page_config(page_title="RAG Test Case Generator", layout="wide")
st.title("Multi-Modal Test Case Generator")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY. Please set it in your .env file.")
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    # Vision Model
    st.subheader("1. Vision Model")
    selected_vision_key = st.selectbox(
        "For Image Description", 
        options=list(VISION_MODELS.keys())
    )
    selected_vision_model = VISION_MODELS[selected_vision_key]
    
    # Generation Model
    st.subheader("2. Generation Model")
    selected_gen_key = st.selectbox(
        "For Test Case Creation",
        options=list(GENERATION_MODELS.keys())
    )
    selected_gen_model = GENERATION_MODELS[selected_gen_key]

    # Guardrail Model
    st.subheader("3. Guardrails")
    selected_guard_key = st.selectbox(
        "Safety Check",
        options=list(GUARDRAIL_MODELS.keys())
    )
    selected_guard_model = GUARDRAIL_MODELS[selected_guard_key]

    st.divider()
    st.header("Knowledge Base")
    
    uploaded_files = st.file_uploader(
        "Upload context files", 
        accept_multiple_files=True,
        type=["pdf", "png", "jpg", "jpeg", "docx", "doc", "txt", "md", "json", "yaml", "xml"]
    )
    
    if st.button("Ingest Files"):
        if not uploaded_files:
            st.warning("Please upload some files first.")
        else:
            with st.spinner(f"Ingesting with {selected_vision_key}..."):
                try:
                    # Reset Data Folder
                    if os.path.exists(DATA_FOLDER):
                        shutil.rmtree(DATA_FOLDER)
                    os.makedirs(DATA_FOLDER)
                    
                    # Save Files
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    st.toast(f"Saved {len(uploaded_files)} files.")

                    # Process
                    ingestion = IngestionEngine(vision_model=selected_vision_model) 
                    retrieval = RetrievalEngine(persist_directory=DB_FOLDER)
                    
                    retrieval.clear() 
                    chunks = ingestion.process_folder(DATA_FOLDER)
                    
                    retrieval.add_documents(chunks)
                    
                    st.success(f"Ingestion Complete! Processed {len(chunks)} chunks.")
                    
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                    logger.error(f"Ingestion failed: {e}")

# Main Area
st.header("Generate Test Cases")

query = st.text_area("Enter your requirement or query:", value="Enter your prompt", height=100)

if st.button("Generate"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner(f"Generating with {selected_gen_key}..."):
            try:
                retrieval = RetrievalEngine(persist_directory=DB_FOLDER)
                generator = GenerationEngine(
                    model_name=selected_gen_model, 
                    guardrail_model=selected_guard_model
                )
                
                docs = retrieval.query(query, top_k=5)
                context_texts = [d.page_content for d in docs]
                
                if not docs:
                    st.warning("No relevant context found. Output might be generic.")
                else:
                    st.info(f"Retrieved {len(docs)} context chunks.")

                result = generator.generate(query, context_texts)
                
                st.subheader("Generated Test Suite")
                
                if result.status == "unsafe":
                    st.error("Request blocked by Guardrails.")
                    st.write("Reason: Input flagged as unsafe.")
                
                elif result.status == "missing_info":
                    st.warning("Missing Information detected.")
                    if result.missing_info_questions:
                         st.write("**Clarification Questions:**")
                         for q in result.missing_info_questions:
                             st.info(f"- {q}")

                elif result.test_cases:
                    for i, uc in enumerate(result.test_cases):
                        with st.expander(f"Case {i+1}: {uc.title} ({uc.type})", expanded=True):
                            st.markdown(f"**Goal**: {uc.goal}")
                            st.markdown(f"**Preconditions**: {uc.preconditions}")
                            if uc.test_data:
                                st.markdown(f"**Test Data**: {uc.test_data}")
                            
                            st.markdown("**Steps**:")
                            for step in uc.steps:
                                st.write(f"- {step}")
                            st.markdown(f"**Expected Results**: {uc.expected_results}")
                            
                            if uc.negative_cases:
                                st.markdown("**Negative Scenarios**:")
                                for neg in uc.negative_cases:
                                    st.write(f"- {neg}")

                            if uc.boundary_cases:
                                st.markdown("**Boundary Cases**:")
                                for boundary in uc.boundary_cases:
                                    st.write(f"- {boundary}")
                elif result.status == "error":
                     st.error(f"Generation failed: {result.missing_info_questions}")

                st.divider()
                with st.expander("Debug: Retrieved Context"):
                    for i, doc in enumerate(docs):
                        score = doc.metadata.get('score', 'N/A')
                        st.caption(f"Source: {doc.metadata.get('source', 'unknown')} | Distance: {score}")
                        st.code(doc.page_content[:600] + "...", language="markdown")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"App error: {e}")
