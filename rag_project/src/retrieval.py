import os
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class RetrievalEngine:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        # Use a high-quality embedding model
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
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
        # Add to Chroma 
        self.vector_store.add_documents(documents)
        print(f"Added {len(documents)} documents to vector store.")

    def query(self, query_text: str, top_k: int = 5) -> List[Document]:
        """
        Retrieves relevant documents based on query.
        """
        return self.vector_store.similarity_search(query_text, k=top_k)

    def clear(self):
        """
        Clears the vector store.
        """
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )

if __name__ == "__main__":
    # Test
    retriever = RetrievalEngine()
    print("Retriever initialized.")
