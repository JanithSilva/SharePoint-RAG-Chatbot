# math_server.py
from mcp.server.fastmcp import FastMCP
from src.services.vector_store import VectorStoreService
from src.settings import load_config
from typing import Dict, Any, List, Annotated


config = load_config()
vector_store = VectorStoreService({ 
            **config["pinecone"],
            **config["openai-embedding"]
        })

mcp = FastMCP("Tools")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool(name="vector_retrieve", description="Retrieve relevant documents from the vector store based on the input query") 
def vector_retrieve(query: str) -> List[str]:
    """
    Retrieve relevant documents from the vector store based on the input query.
    
    This function searches the vector store for documents that are semantically similar
    to the input query and returns the top matching documents.

    Args:
        query (str): The search query string used to retrieve relevant documents.

    Returns:
        List[str]: A list of the top matching document contents (as strings)
    """
    documents = vector_store.retrieve(query, top_k=5)
    # Write retrieved documents to documents key in state
    return  documents



if __name__ == "__main__":
    mcp.run(transport="stdio")