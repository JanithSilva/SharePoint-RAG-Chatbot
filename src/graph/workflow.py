from langgraph.graph import Graph
from .nodes import retrieve_node, query_graph_node, generate_response_node

def create_workflow():
    workflow = Graph()
    
    # Add nodes for each step
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("query_graph", query_graph_node)
    workflow.add_node("generate", generate_response_node)
    
    # Define the flow between nodes
    workflow.add_edge("retrieve", "query_graph")
    workflow.add_edge("query_graph", "generate")
    
    # Set the entry point
    workflow.set_entry_point("retrieve")
    return workflow.compile()