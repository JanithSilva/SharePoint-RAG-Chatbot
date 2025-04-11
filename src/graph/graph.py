from langchain_openai import AzureChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import tools_condition, ToolNode
from dotenv import load_dotenv
from typing import Dict, Any, List, Annotated
import os
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
import operator
from typing_extensions import TypedDict
from src.graph.nodes import (retrieve, generate)

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents
    entities: List[dict]  # List of entities

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("generate", generate)  # generate

workflow.add_edge(START, "retrieve")  # Start with retrieve
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# builder = StateGraph(MessagesState)
# builder.add_node("assistant", assistant)
# builder.add_node("tools", ToolNode(tools))
# builder.add_edge(START, "assistant")
# builder.add_conditional_edges(
#     "assistant",
#     # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
#     # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
#     tools_condition,
# )
# builder.add_edge("tools", "assistant")


# Compile graph
graph = workflow.compile()


