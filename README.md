# 🦜🕸️ LangGraph RAG: Agentic Retrieval Augmented Generation

An advanced RAG system leveraging **LangGraph** for stateful agentic workflows, **FastAPI** for robust backend services, and **Streamlit** for an interactive frontend. This project features a decoupled architecture designed for PDF ingestion, vectorization, and intelligent query answering using Groq.

## 📂 Project Structure

```text
langgraph-rag/
├── backend/
│   ├── packages/
│   │   ├── graph.py         # 🧠 State Machine & Workflow Logic (LangGraph)
│   │   ├── rag_nodes.py     # ⚙️ Individual Agent Functions (Retrieve, Grade, Generate)
│   │   └── ingestion.py     # 📄 PDF Processing & Vectorization pipeline
│   ├── server.py            # 🚀 FastAPI Entry Point
│   └── .env                 # 🔒 Environment Variables
├── frontend/
│   └── app.py               # 💻 Streamlit User Interface
└── requirements.txt         # 📦 Dependencies
```

## ⚡ Setup & Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/Visheshsing/RAG_BASED_CHATBOT_FOR_PDFs](https://github.com/Visheshsing/RAG_BASED_CHATBOT_FOR_PDFs.git)
cd agentic-rag
```

### 2. Create Virtual Environment

Create and activate an isolated Python environment to manage dependencies.

**Linux/MacOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the `backend/` directory and add your API credentials.

```bash
# backend/.env
GROQ_API_KEY=gsk_your_api_key_here
```

## 🏃‍♂️ Execution Guide

This system uses a decoupled architecture. You must run the API Server and Frontend Client in separate terminals.

### Terminal 1: Backend Server

Starts the FastAPI server containing the Agentic Graph.

```bash
python backend/server.py
```
*Server runs at: `http://localhost:8000`* *Swagger Docs: `http://localhost:8000/docs`*

### Terminal 2: Frontend UI

Launches the Streamlit interface.

```bash
streamlit run frontend/app.py
```
*UI runs at: `http://localhost:8501` (default)*

## 📡 API Endpoints & Testing

The API exposes three main endpoints. You can test these using **Postman**, **cURL**, or the **Swagger UI**.

### 1️⃣ Health Check

Verifies the API is online.
- **Method:** `GET`
- **URL:** `http://localhost:8000/`

**Response:**
```json
{
    "status": "active",
    "service": "RAG API"
}
```

### 2️⃣ Upload Document (Background Task)

Uploads a PDF and queues it for ingestion. Files are renamed using the `user_id` to prevent conflicts.

- **Method:** `POST`
- **URL:** `http://localhost:8000/upload`
- **Content-Type:** `multipart/form-data`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file` | File | The .pdf document you want to ingest. |
| `user_id` | Text | Unique identifier (e.g., `user_123`) for file isolation. |

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/your/document.pdf" \
  -F "user_id=user_123"
```

**Response:**
```json
{
    "message": "Upload successful! Processing started in the background.",
    "filename": "original_name.pdf",
    "user_id": "test_user"
}
```

### 3️⃣ Chat / RAG Inference

Ask questions about the uploaded document.

- **Method:** `POST`
- **URL:** `http://localhost:8000/chat`
- **Content-Type:** `application/json`

**Payload:**
```json
{
  "query": "What are the main findings in the document?",
  "thread_id": "session_v1",
  "user_id": "test_user"
}
```
> ⚠️ **Critical:** The `user_id` must match the one used in the `/upload` step.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "Summarize the key points.",
        "thread_id": "thread_abc",
        "user_id": "user_123"
      }'
```

## ⚠️ Troubleshooting

| Error | Code | Cause | Fix |
| :--- | :--- | :--- | :--- |
| **Only PDF files are allowed** | 400 | Uploaded file is not a `.pdf` | Ensure the file extension is `.pdf` |
| **Internal Server Error** | 500 | Ingestion failed or API key missing | Check server terminal logs. Ensure `.env` has API keys. |
| **Field required** | 422 | Missing `user_id` or `file` | Check Postman body/payload structure. |

## 🔮 Future Roadmap

- [ ] **Multi-Document Support:** Ingest multiple PDFs simultaneously.
- [ ] **Hybrid Search:** Combine Keyword (BM25) with Semantic Search for better retrieval.
- [ ] **Streaming:** Stream tokens to the UI for a better User Experience.
- [ ] **Dockerization:** Containerize the application for easy deployment.
