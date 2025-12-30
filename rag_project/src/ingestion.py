import os
import glob
import hashlib
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
import pytesseract
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Tesseract Configuration ---
# pytesseract is just a wrapper; it needs to know where the actual tesseract.exe is.
# On Windows, it's not always in the PATH, so we explicitly set it.
import platform
if platform.system() == "Windows":
    # Check common default install location
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Also check if user set a specific env var (overrides default)
    env_path = os.getenv("TESSERACT_CMD")
    if env_path and os.path.exists(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path

class IngestionEngine:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_folder(self, folder_path: str) -> List[Document]:
        """
        Iterates through a folder and processes all supported files with robust error handling.
        """
        documents = []
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
                
                # Add content hash for deduplication/quality checks later
                for doc in raw_docs:
                    doc.metadata["file_hash"] = self._compute_hash(doc.page_content)
                    documents.append(doc)

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        # Chunking
        chunked_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Generated {len(chunked_docs)} chunks from {len(documents)} source docs.")
        return chunked_docs

    def _process_pdf(self, file_path: str) -> List[Document]:
        """
        Processes PDF. Falls back to OCR if standard loader returns empty text (scanned PDFs).
        """
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Check if text was actually extracted
            total_text_len = sum(len(d.page_content.strip()) for d in docs)
            
            # If PDF text is empty it can be a scanned image-only PDF
            if total_text_len < 50: 
                logger.warning(f"PDF {os.path.basename(file_path)} seems to be scanned. Attempting OCR...")
                return self._pdf_ocr_fallback(file_path)
                
            return docs
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return []

    def _process_text(self, file_path: str) -> List[Document]:
        """
        Tries multiple encodings to prevent crashes on non-UTF-8 files.
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

    def _process_image(self, file_path: str) -> List[Document]:
        """
        Extracts text from images using OCR.
        """
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                logger.warning(f"No text found in image {file_path}")
                return []
            
            doc = Document(
                page_content=f"--- IMAGE START: {os.path.basename(file_path)} ---\n{text}\n--- IMAGE END ---",
                metadata={"source": file_path, "type": "image"}
            )
            return [doc]
        except Exception as e:
            logger.warning(f"OCR failed for {file_path}. Is Tesseract installed? Error: {e}")
            return []

    def _pdf_ocr_fallback(self, file_path: str) -> List[Document]:
        """
        Fallback: Converts PDF pages to images and runs OCR.
        Requires 'pdf2image' library.
        """
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(file_path)
            full_text = ""
            for i, page in enumerate(pages):
                text = pytesseract.image_to_string(page)
                full_text += f"\n--- Page {i+1} ---\n{text}"
            
            return [Document(
                page_content=full_text, 
                metadata={"source": file_path, "type": "scanned_pdf"}
            )]
        except ImportError:
            logger.warning("pdf2image not installed. Skipping OCR fallback for PDF.")
            return []
        except Exception as e:
            logger.error(f"PDF OCR fallback failed: {e}")
            return []

    def _process_docx(self, file_path: str) -> List[Document]:
        """
        Processes Word Documents (.docx).
        """
        try:
            # First try Docx2txtLoader (faster, strictly for .docx)
            if file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                return loader.load()
            
            # Fallback (or for .doc) - Unstructured
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error reading Word Doc {file_path}: {e}")
            return []

    def _compute_hash(self, content: str) -> str:
        return hashlib.md5(content.encode('utf-8')).hexdigest()

if __name__ == "__main__":
    engine = IngestionEngine()
    if os.path.exists("rag_project/data"):
        docs = engine.process_folder("rag_project/data")