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
from src.agents.RAG_chatbot.nodes import (vector_retrieve, generate,determine_output,grade_documents,decide_to_generate, grade_generation_v_documents_and_question)  

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    input: str #User Question
    generation: str  # LLM generation
    documents: List[str]  
    error: Optional[str] 
    output: str  
    loop_step: Annotated[int, operator.add]
    max_retries: int  # Max number of retries for answer generation

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
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": "determine_output",
        "not useful": "generate",
        "max retries":  "determine_output",
    },
)
workflow.add_edge("determine_output", END)

graph = workflow.compile()


