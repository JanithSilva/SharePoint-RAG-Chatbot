from typing import Dict
from src.services.vector_store import VectorStoreService
from src.services.graph_store import GraphStoreService
from langchain_openai import AzureChatOpenAI
from src.settings import load_config
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

config = load_config()
vector_store = VectorStoreService({ 
            **config["pinecone"],
            **config["openai-embedding"]
        })
graph_service = GraphStoreService()
openai_config = config["openai-llm"]
llm = AzureChatOpenAI(
            azure_deployment=openai_config["azure_deployment"],
            openai_api_version=openai_config["api_version"],
            azure_endpoint=openai_config["azure_endpoint"],
            api_key=openai_config["api_key"]
        )

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    documents = vector_store.retrieve(question, top_k=3)
    # Write retrieved documents to documents key in state
    return {"documents": documents}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    def format_docs(docs):
        return "\n\n".join(doc for doc in docs)
    
    rag_prompt = """You are an assistant for question-answering tasks. 

    Here is the context to use to answer the question:
        
    {context} 

    Think carefully about the above context. 

    Now, review the user question:

    {question}

    Provide an answer to this questions using only the above context. 

    Answer:"""

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1, "question": question, "documents": documents}

# def generate_response_node(state: Dict) -> Dict:
#     """Generate a response using the language model."""
#     documents = state.get("documents", [])
#     entities = state.get("entities", [])
#     context = "\n".join(documents) + "\n" + "\n".join([str(e) for e in entities])
#     response = graph_service.llm.generate_response(context, state["user_input"])
#     state["response"] = response
#     return state

# def query_graph_node(state: Dict) -> Dict:
#     """Query the knowledge graph for entities."""
#     query = state["user_input"]
#     entities = graph_service.query_entities(query, limit=5)
#     state["entities"] = entities
#     return state
