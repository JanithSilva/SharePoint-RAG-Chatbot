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
            api_key=openai_config["api_key"],
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
        state (dict): Filtered out irrelevant documents.
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
        error = "No relevant documents found."
    else:
        error = None

    return {
        "documents": filtered_documents,
        "error": error,
    }

    
def determine_output(state):
    """
    Determines the final output of the agent.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered output state.
    """

    error = state.get("error", False)
    max_retries = state.get("max_retries", 3)
    generation = state.get("generation", None)
    loop_step = state.get("loop_step", 0)

    if error:
        output = error
    elif loop_step >= max_retries:
        output = "Max retries reached."
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

    if error is not None:
        return "determine_output"
    else:
        # We have relevant documents, so generate answer
        return "generate"
    

def grade_generation_v_documents_and_question(state):
    """
    
    """
    #Get state variables
    question = state["input"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)

    #hasulination check propmts
    hallucination_grader_instructions = """

        You are a teacher grading a quiz. 

        You will be given FACTS and a STUDENT ANSWER. 

        Here is the grade criteria to follow:

        (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

        (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

        Score:

        A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

        A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset."""
    
    
    hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

        Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""
    
    #Question-answering check prompts
    answer_grader_instructions = """You are a teacher grading a quiz. 

        You will be given a QUESTION and a STUDENT ANSWER. 

        Here is the grade criteria to follow:

        (1) The STUDENT ANSWER helps to answer the QUESTION

        Score:

        A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

        The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

        A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset."""

    answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""
    


    #execute hallucination check
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format( documents=format_docs(documents), generation=generation.content)

    result = llm.invoke([SystemMessage(content=hallucination_grader_instructions)] + [HumanMessage(content=hallucination_grader_prompt_formatted)])

    grade = json.loads(result.content)["binary_score"]
    
    #Check hallucination
    if grade == "yes":
        #No hallucination, so check question-answering
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]

        if grade == "yes":
            #No hallucination, and question answered, so return useful
            return "useful"
        elif state["loop_step"] <= max_retries:
            #No hallucination, but question not answered, so retry
            return "not useful"
        else:
            #No hallucination, but question not answered, and max retries reached, so return max retries
            return "max retries"
    elif state["loop_step"] <= max_retries:
        #Hallucination detected, so retry
        return "not supported"
    else:
        #Hallucination detected, and max retries reached, so return max retries
        return "max retries"