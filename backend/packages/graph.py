import operator
from typing import TypedDict, List, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage

# Import nodes from your rag_nodes.py file
# Ensure these function names exist in your rag_nodes.py
from .rag_nodes import (
    retrieve, 
    grade_documents, 
    generate, 
    transform_query,
    decide_next_step,
    increment_loop,
    finalize_failure
)

class AgentState(TypedDict):
    """
    Represents the state of our graph.
    """
    # 'messages' accumulates conversation history (Human + AI messages)
    messages: Annotated[List[BaseMessage], operator.add]
    # The current user question
    question: str
    # Documents retrieved from ChromaDB
    documents: List[Document]
    # Flag to track if the grader found relevant info
    relevant_found: bool
    # Counter to prevent infinite search loops
    loop_step: int
    # ✅ FINAL ANSWER: The string content to be sent to the API
    answer: str 

def build_graph():
    """
    Constructs the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # --- 1. Define Nodes ---
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("increment_loop", increment_loop)
    workflow.add_node("finalize_failure", finalize_failure)

    # --- 2. Define Connectivity (Edges) ---
    
    # Starting point
    workflow.add_edge(START, "retrieve")
    
    # After retrieval, always grade the documents
    workflow.add_edge("retrieve", "grade_documents")
    
    # ⚡ CONDITIONAL LOGIC: 
    # Based on the grade, we either generate, retry (transform), or fail.
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {
            "generate": "generate",
            "transform_query": "increment_loop",
            "finalize_failure": "finalize_failure"
        },
    )
    
    # Retry Loop: Increment counter -> Rewrite Query -> Retrieve again
    workflow.add_edge("increment_loop", "transform_query")
    workflow.add_edge("transform_query", "retrieve")
    
    # Terminal Edges (The end of the process)
    workflow.add_edge("generate", END)
    workflow.add_edge("finalize_failure", END)

    # --- 3. Configuration & Compilation ---
    
    # MemorySaver allows the graph to remember previous interactions via thread_id
    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)

# Create the executable graph instance
app_graph = build_graph()