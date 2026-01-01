import os
import sys
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.logger import setup_logger
from core.config import EMBEDDING_MODEL_NAME

logger = setup_logger()

class RetrievalEngine:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents: List[Document]):
        """
        Adds documents to the vector store.
        """
        if not documents:
            return
        
        self.vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to vector store.")

    def query(self, query_text: str, top_k: int = 5, score_threshold: float = 1.2) -> List[Document]:
        """
        Retrieves relevant documents. 
        Note: For L2 distance (default in Chroma), LOWER is better.
        The score is a distance. 
        """
        results_with_scores = self.vector_store.similarity_search_with_score(query_text, k=top_k)
        
        valid_docs = []
        for doc, score in results_with_scores:
            logger.info(f"Retrieved doc '{doc.metadata.get('source', 'unknown')}' with distance: {score}")
            
            # Distance threshold: If distance is too high (poor match), filter it.
            # all-MiniLM-L6-v2 distances usually range 0-2. 0 is exact.
            if score < score_threshold: 
               valid_docs.append(doc)
            else:
                logger.warning(f"Filtered out doc due to low confidence (high distance): {score}")

        if not valid_docs:
            logger.info("No documents passed the confidence threshold.")
            return []
            
        return valid_docs

    def clear(self):
        """
        Clears the vector store.
        """
        try:
            self.vector_store.delete_collection()
            # Re-initialize
            self.vector_store = Chroma(
                collection_name="rag_collection",
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            logger.info("Vector store cleared.")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
