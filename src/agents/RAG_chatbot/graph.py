from langchain_openai import AzureChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import tools_condition, ToolNode
from dotenv import load_dotenv
from typing import Dict, Any, List, Annotated, TypedDict, Optional
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
import operator
from typing_extensions import TypedDict
from src.agents.RAG_chatbot.nodes import (vector_retrieve, generate,determine_output,grade_documents,decide_to_generate)  

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    input: str
    generation: str  
    documents: List[str]  
    error: Optional[str] 
    error_message: Optional[str]  
    output: str  
    entities:str

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", vector_retrieve)
workflow.add_node("generate", generate)  
workflow.add_node("determine_output", determine_output)  
workflow.add_node("grade_documents", grade_documents)  

workflow.add_edge(START, "retrieve")  
workflow.add_edge("retrieve", "grade_documents")  
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "determine_output": "determine_output",
        "generate": "generate",
    },
)
workflow.add_edge("generate", "determine_output")
workflow.add_edge("determine_output", END)

graph = workflow.compile()


