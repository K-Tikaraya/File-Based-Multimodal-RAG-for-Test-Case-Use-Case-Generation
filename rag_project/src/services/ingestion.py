import os
import glob
import hashlib
import base64
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from groq import Groq
import pymupdf4llm
import sys

# Add parent directory to path to allow imports from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.logger import setup_logger
from core.config import GROQ_API_KEY

# Initialize Logger
logger = setup_logger()

class IngestionEngine:
    def __init__(self, chunk_size=1000, chunk_overlap=200, vision_model="meta-llama/llama-4-maverick-17b-128e-instruct"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vision_model = vision_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Groq Client for Vision
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not found. Vision capabilities will be disabled.")
            self.groq_client = None
        else:
            self.groq_client = Groq(api_key=GROQ_API_KEY)

    def process_folder(self, folder_path: str) -> List[Document]:
        """
        Iterates through a folder and processes files (PDF, Images, Text, Word)
        """
        documents = []
        # Ensure folder exists
        if not os.path.exists(folder_path):
             logger.warning(f"Folder {folder_path} does not exist.")
             return []

        files = glob.glob(os.path.join(folder_path, "**", "*.*"), recursive=True)
        
        logger.info(f"Found {len(files)} files in {folder_path}")
        
        for file_path in files:
            ext = os.path.splitext(file_path)[1].lower()
            try:
                raw_docs = []
                if ext == ".pdf":
                    raw_docs = self._process_pdf(file_path)
                elif ext in [".docx", ".doc"]:
                    raw_docs = self._process_docx(file_path)
                elif ext in [".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".log"]:
                    raw_docs = self._process_text(file_path)
                elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
                    raw_docs = self._process_image(file_path)
                
                # Deduplication Hash
                for doc in raw_docs:
                    doc.metadata["file_hash"] = self._compute_hash(doc.page_content)
                    documents.append(doc)

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        # Chunking
        if not documents:
            return []
            
        chunked_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Generated {len(chunked_docs)} chunks from {len(documents)} source docs.")
        return chunked_docs

    def _process_image(self, file_path: str) -> List[Document]:
        """
        Sends image to Groq Vision model for description.
        """
        if not self.groq_client:
            logger.warning(f"Skipping image {file_path} - No Groq Client")
            return []

        try:
            # Encode image
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            completion = self.groq_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Describe this UI screenshot or diagram in technical detail for a QA engineer. List all visible buttons, fields, error messages, and layout elements."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            
            description = completion.choices[0].message.content
            content = f"[IMAGE SOURCE: {os.path.basename(file_path)}]\nDescription: {description}"
            
            return [Document(
                page_content=content,
                metadata={"source": os.path.basename(file_path), "page": 1, "type": "image"}
            )]

        except Exception as e:
            logger.error(f"Vision API handling failed for {file_path}: {e}")
            return []

    def _process_pdf(self, file_path: str) -> List[Document]:
        """
        Uses pymupdf4llm to convert PDF to Markdown. 
        """
        try:
            md_text = pymupdf4llm.to_markdown(file_path)
            return [Document(
                page_content=md_text,
                metadata={"source": os.path.basename(file_path), "page": 0, "type": "pdf_markdown"}
            )]
        except Exception as e:
            logger.error(f"PDF processing failed for {file_path}: {e}")
            return []

    def _process_text(self, file_path: str) -> List[Document]:
        """
        Tries multiple encodings.
        """
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                loader = TextLoader(file_path, encoding=enc)
                return loader.load()
            except Exception:
                continue
        
        logger.error(f"Could not decode text file {file_path} with supported encodings.")
        return []

    def _process_docx(self, file_path: str) -> List[Document]:
        """
        Processes Word Documents (.docx).
        """
        try:
            if file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                return loader.load()
            
            # Fallback for .doc
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error reading Word Doc {file_path}: {e}")
            return []

    def _compute_hash(self, content: str) -> str:
        return hashlib.md5(content.encode('utf-8')).hexdigest()
