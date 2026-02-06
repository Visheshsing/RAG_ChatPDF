import os
import shutil
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware # <--- NEW: Standard Security
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from packages.graph import app_graph
from packages.ingestion import ingest_data 

app = FastAPI(title="Professional RAG API")

# --- 1. CONFIGURATION ---
# Add CORS to allow requests from your Streamlit app or other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True) 

# --- 2. DATA MODELS ---
class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default_thread"
    user_id: str # Required for user isolation

class ChatResponse(BaseModel):
    answer: str
    context: list

# --- 3. ENDPOINTS ---

@app.get("/")
async def health_check():
    """Simple check to see if server is running."""
    return {"status": "active", "service": "RAG API"}

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...) # Recieve user_id as Form Data
):
    try:
        # Validate File
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
        # Sanitize filename (prevent overwrites between users)
        safe_filename = f"{user_id}_{os.path.basename(file.filename)}"
        file_path = os.path.join(DATA_DIR, safe_filename)
        
        # Save File
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"✅ File saved for user {user_id}: {file_path}")

        # Trigger Background Task
        # Must match ingest_data(file_path, user_id)
        background_tasks.add_task(ingest_data, file_path, user_id)
        
        return {
            "message": "Upload successful! Processing started in the background.", 
            "filename": file.filename,
            "user_id": user_id
        }
            
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Pass User ID to Graph Config
        config = {
            "configurable": {
                "thread_id": request.thread_id,
                "user_id": request.user_id
            }
        }
        
        inputs = {
            "messages": [HumanMessage(content=request.query)],
            "question": request.query 
        }
        
        # Invoke Graph
        result = app_graph.invoke(inputs, config=config, recursion_limit=15)
        
        context_text = [doc.page_content for doc in result.get("documents", [])]
        
        return ChatResponse(
            answer=result.get("answer", "No answer generated."),
            context=context_text
        )
    except Exception as e:
        print(f"❌ Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Ensure this runs on 0.0.0.0 so it's accessible externally if needed
    uvicorn.run(app, host="0.0.0.0", port=8000)