from typing import Dict
from src.services.vector_store import VectorStoreService
from src.services.graph_store import GraphStoreService
from langchain_openai import AzureChatOpenAI
from src.settings import load_config
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

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

def format_docs(docs):
        return "\n\n".join(doc for doc in docs)

def vector_retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    input = state["input"]
    documents = vector_store.retrieve(input, top_k=5)
    return {"documents": documents}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["input"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    
    rag_prompt = """You are an assistant for question-answering tasks. 

    Here is the context to use to answer the question:
        
    {context} 

    Think carefully about the above context. 

    Now, review the user question:

    {question}

    Provide an answer to this questions using only the above context. 

    Answer:"""

    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to fallback.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated fallback state.
    """

    doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""
    
    doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

        This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

        Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""
    

    input = state.get("input", "")
    documents = state.get("documents", [])

    filtered_documents = []
    for doc in documents:
        prompt = doc_grader_prompt.format(document=doc, question=input)
        result = llm.invoke(
            [SystemMessage(content=doc_grader_instructions),
             HumanMessage(content=prompt)]
        )
        
        try:
            score = json.loads(result.content)["binary_score"]
            if score == "yes":
                filtered_documents.append(doc)
        except (KeyError, ValueError, json.JSONDecodeError):
            score = 0  # Fail-safe

    if  len(filtered_documents) < 1:
        error = True
    else:
        error = False

    return {
        "documents": filtered_documents,
        "error": error,
        "error_message": "No relevant documents found." if error else None,
    }

    
def determine_output(state):
    """
    Determines the final output of the agent.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out ouput state.
    """

    error = state.get("error", False)
    error_message = state.get("error_message", None)
    generation = state.get("generation", None)

    if error:
        output = error_message
    else:
        output = generation       

    return {
        "output": output
        }

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or end execution

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    error = state.get("error", False)

    if error == True:
        return "determine_output"
    else:
        # We have relevant documents, so generate answer
        return "generate"
    

def route_question(state):
    """
    Routes the user question to either the vectorstore (for semantic search on document content)
    or the tool-calling agent (for entity relationships or file metadata).

    Args:
        state (dict): Current state containing the user question.

    Returns:
        str: The name of the next node in the graph ("vectorstore" or "tools").
    """
    router_instructions = """You are an intelligent routing agent.

    Determine whether the user's question should be handled by:
    - The vectorstore: for questions about document **content**, semantics, or extracted text.
    - The tool-calling agent: for questions about **entities**, **relationships**, or **file metadata** (e.g., file name, modified date).

    Respond with a JSON object using a single key:
    { "datasource": "vectorstore" } or { "datasource": "tools" } — choose only one based on the question."""

    try:
        response = llm.invoke(
            [SystemMessage(content=router_instructions)] +
            [HumanMessage(content=state["input"][-1]["content"])]
        )
        data = json.loads(response.content.strip())
        source = data.get("datasource", "").lower()

        if source == "vectorstore":
            return "vectorstore"
        elif source == "tools":
            return "tools"
        else:
            # Fallback 
            return "vectorstore" 

    except Exception as e:
        return "vectorstore"  # Safe fallback