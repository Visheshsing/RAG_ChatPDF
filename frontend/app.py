import streamlit as st
import requests
import uuid

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8000"
CHAT_URL = f"{BASE_URL}/chat"
UPLOAD_URL = f"{BASE_URL}/upload"

st.set_page_config(page_title="Chat with your PDFs", page_icon="🧠", layout="wide")

# --- STATE MANAGEMENT ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agentic RAG")
    st.divider()

    # 1. User Identity (Crucial for Multi-User Backend)
    st.markdown("### 👤 User Profile")
    st.session_state.user_id = st.text_input(
        "Enter User ID", 
        value=st.session_state.user_id,
        help="Unique ID to separate your documents from others."
    )
    
    st.divider()
    
    # 2. Upload Section
    st.markdown("### 📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF to chat with:", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Ingest File", type="primary"):
            if not st.session_state.user_id:
                st.error("⚠️ Please enter a User ID first!")
            else:
                with st.spinner("🚀 Uploading and Indexing..."):
                    try:
                        # Prepare payload for Multipart/Form-Data
                        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                        # Send user_id as form data
                        data = {"user_id": st.session_state.user_id}
                        
                        response = requests.post(UPLOAD_URL, files=files, data=data, timeout=30)
                        
                        if response.status_code == 200:
                            st.success("✅ File Ingested Successfully!")
                            st.session_state.messages = [] # Clear history on new file
                        else:
                            st.error(f"Upload failed: {response.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
    
    st.divider()
    
    # 3. Controls
    st.markdown("### ⚙️ Controls")
    if st.button("🧹 Clear Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("💬 Chat with your PDF")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show sources if available
        if "context" in msg and msg["context"]:
            with st.expander("📚 View Retrieved Sources"):
                for i, doc in enumerate(msg["context"]):
                    st.caption(f"**Source {i+1}:**")
                    st.markdown(f"> {doc}")
                    st.divider()

# Handle Input
if prompt := st.chat_input("Ask a question..."):
    # 1. Validation
    if not st.session_state.user_id:
        st.warning("⚠️ Please set a User ID in the sidebar to start chatting.")
        st.stop()

    # 2. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Get Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare JSON Payload
                payload = {
                    "query": prompt, 
                    "thread_id": st.session_state.thread_id,
                    "user_id": st.session_state.user_id # <--- Pass User ID
                }
                
                response = requests.post(CHAT_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    context = data.get("context", [])
                    
                    st.markdown(answer)
                    
                    # Store in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "context": context
                    })
                    
                    # Show sources immediately
                    if context:
                        with st.expander("📚 View Retrieved Sources"):
                            for i, doc in enumerate(context):
                                st.caption(f"**Source {i+1}:**")
                                st.markdown(f"> {doc}")
                                st.divider()
                else:
                    st.error(f"❌ API Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the backend server. Is it running?")
            except Exception as e:
                st.error(f"❌ Error: {e}")