import os
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig # <--- Fix for Warning
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "../chroma_db")

# --- CONFIGURATION ---
llm_fast = ChatGroq(model="llama-3.3-70b-versatile", temperature=0) 
llm_smart = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- DATA MODELS ---

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevant to the question? 'yes' or 'no'")

class HallucinationGrade(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts? 'yes' or 'no'")

# --- MAIN NODES ---

def retrieve(state: Dict[str, Any], config: RunnableConfig): # <--- Fixed Type Hint
    """
    Retrieves documents based on the current user's ID.
    """
    print("---RETRIEVE: Fetching Context---")
    
    # 1. Get User ID from Config
    config_dict = config.get("configurable", {})
    user_id = config_dict.get("user_id")
    
    if not user_id:
        print("⚠️ Warning: No user_id found. Searching all documents.")
    
    # 2. Extract Question
    question = state.get("question", state["messages"][-1].content if "messages" in state else "")

    # 3. Dynamic Vector Store Connection
    vectorstore = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )
    
    # 4. Filtered Search
    search_kwargs = {"k": 5}
    if user_id:
        search_kwargs["filter"] = {"user_id": user_id}
        
    documents = vectorstore.similarity_search(question, **search_kwargs)
    
    return {"documents": documents, "question": question}

def grade_documents(state: Dict[str, Any]):
    print("---GRADE: Filtering Irrelevant Docs---")
    question = state["question"]
    documents = state["documents"]
    
    structured_llm_grader = llm_fast.with_structured_output(GradeDocuments)
    
    system = """You are a strict grader checking if a document is relevant to the user question.
    If the document contains keywords or semantic meaning related to the question, grade it as 'yes'.
    Otherwise, grade it as 'no'."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    grader_chain = grade_prompt | structured_llm_grader
    
    filtered_docs = []
    relevant_found = False

    for d in documents:
        try:
            score = grader_chain.invoke({"question": question, "document": d.page_content})
            if score.binary_score.lower() == "yes":
                filtered_docs.append(d)
                relevant_found = True
        except Exception as e:
            print(f"⚠️ Grading Error: {e}")
            continue
            
    return {"documents": filtered_docs, "question": question, "relevant_found": relevant_found}

def transform_query(state: Dict[str, Any]):
    print("---TRANSFORM: Optimizing Query---")
    question = state["question"]
    
    msg = [
        ("system", "You are an expert at refining search queries for vector databases."),
        ("human", f"The original query returned no results. \n Original: {question} \n Write a better, more specific query to find relevant info. Output ONLY the query.")
    ]
    better_question = llm_fast.invoke(msg).content
    
    return {"question": better_question}

def generate(state: Dict[str, Any]):
    print("---GENERATE: Drafting Answer---")
    documents = state["documents"]
    messages = state["messages"]
    
    if not documents:
        return {
            "answer": "I couldn't find relevant information in your uploaded documents.",
            "messages": [AIMessage(content="I couldn't find relevant information in your uploaded documents.")]
        }

    context_text = "\n\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(documents)])
    
    system_msg = SystemMessage(content=f"""You are a helpful assistant for Question-Answering tasks.
    
    INSTRUCTIONS:
    1. Use ONLY the following context to answer the question.
    2. If you don't know the answer, say "I don't know."
    3. Keep the answer concise and professional.
    4. Cite the source number (e.g. [Source 1]) for key facts.
    
    CONTEXT:
    {context_text}
    """)
    
    response = llm_smart.invoke([system_msg] + messages)
    generated_answer = response.content

    # --- GUARDRAIL ---
    print("---GUARDRAIL: Checking for Hallucinations---")
    
    hallucination_grader = llm_smart.with_structured_output(HallucinationGrade)
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", "Compare the answer to the facts. If the answer is supported by the facts, say 'yes'. If it contains outside info or hallucinations, say 'no'."),
        ("human", "Facts: \n\n {documents} \n\n Answer: {generation}")
    ])
    
    hallucination_chain = hallucination_prompt | hallucination_grader
    
    try:
        grade = hallucination_chain.invoke({"documents": context_text, "generation": generated_answer})
        if grade.binary_score.lower() == "no":
            print("🚨 HALLUCINATION DETECTED!")
            final_answer = "I found some documents, but I cannot confidently answer based *only* on them."
        else:
            final_answer = generated_answer
    except Exception as e:
        print(f"⚠️ Guardrail Skipped: {e}")
        final_answer = generated_answer

    return {
        "answer": final_answer, 
        "messages": [AIMessage(content=final_answer)]
    }

# --- LOGIC / HELPER NODES (NEW) ---

def increment_loop(state: Dict[str, Any]):
    """Increments the retry counter."""
    current_step = state.get("loop_step", 0)
    return {"loop_step": current_step + 1}

def decide_next_step(state: Dict[str, Any]):
    """Determines if we should Generate, Retry, or Give Up."""
    relevant = state.get("relevant_found", False)
    step = state.get("loop_step", 0)

    if relevant:
        print("---DECISION: DOCS FOUND -> GENERATE---")
        return "generate"
    
    elif step <= 3: # Max 3 retries
        print(f"---DECISION: NO DOCS (Attempt {step}/3) -> OPTIMIZE QUERY---")
        return "transform_query"
    
    else:
        print("---DECISION: MAX RETRIES REACHED -> STOPPING---")
        return "finalize_failure"

def finalize_failure(state: Dict[str, Any]):
    """Returns a polite failure message."""
    return {
        "answer": "I apologized, but after multiple searches, I could not find relevant information in the uploaded document.",
        "messages": [AIMessage(content="I could not find relevant information after multiple attempts.")]
    }