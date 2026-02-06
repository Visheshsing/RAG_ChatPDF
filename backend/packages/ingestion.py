import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
DB_PATH = os.path.join(BASE_DIR, "../chroma_db")

def ingest_data(file_path, user_id):
    """
    Ingests a specific PDF file using Optimized Recursive Splitting.
    Tags data with user_id for multi-user isolation.
    """
    if not file_path or not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    print(f"📥 Loading PDF for User [{user_id}]: {file_path}")

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return False

    if not docs:
        print("❌ PDF appears empty.")
        return False

    # --- ⚡ OPTIMIZED SPLITTING STRATEGY ---
    print("🔪 Chunking text with Optimized Strategy...")
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        
        # 1. Larger Chunk Size: 
        # Captures more context (approx 3-4 paragraphs). 
        # Helps the LLM understand the "why" behind the text.
        chunk_size=1000, 
        
        # 2. Larger Overlap: 
        # Prevents cutting vital info at the borders. 
        # Ensures smooth transitions between chunks.
        chunk_overlap=200, 
        
        # 3. Smart Separators:
        # Priority order: Double Newline (Paragraphs) -> Newline -> Sentences -> Words
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    splits = text_splitter.split_documents(docs)

    # 4. Add User Metadata
    print(f"🏷️ Tagging {len(splits)} chunks for user: {user_id}")
    for split in splits:
        split.metadata["user_id"] = user_id
        split.metadata["source"] = os.path.basename(file_path)

    # 5. Embed & Save (Append Mode)
    print(f"💾 Appending to Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print("✅ Ingestion Complete!")
    return True

if __name__ == "__main__":
    # Test
    test_file = os.path.join(DATA_DIR, "sample.pdf")
    if os.path.exists(test_file):
        ingest_data(test_file, user_id="test_user")