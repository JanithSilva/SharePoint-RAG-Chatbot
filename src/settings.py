import os
from dotenv import load_dotenv
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    load_dotenv()
    return {
        "sharepoint": {
            "url": os.getenv("SHAREPOINT_SITE_URL"),
            "username": os.getenv("SHAREPOINT_USERNAME"),
            "password": os.getenv("SHAREPOINT_PASSWORD"),
            "library_name": os.getenv("SHAREPOINT_LIBRARY_NAME"),
        },
        "azure_doc_intel": {
            "key": os.getenv("AZURE_DOCUMENT_INTEL_KEY"),
            "endpoint": os.getenv("AZURE_DOCUMENT_INTEL_ENDPOINT")
        },
        "pinecone": {
            "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
            "environment": os.getenv("PINECONE_ENVIRONMENT"),
            "index_name": "sharepoint-docs"

        },
        "neo4j": {
            "uri": os.getenv("NEO4J_URI"),
            "user": os.getenv("NEO4J_USER"),
            "password": os.getenv("NEO4J_PASSWORD")
        },
        "openai-embedding": {
            "azure_endpoint": os.getenv("EMBEDDING_MODEL_ENDPOINT"),
            "api_key": os.getenv("EMBEDDING_MODEL_KEY"),
            "api_version": os.getenv("EMBEDDING_MODEL_API_VERSION"),
            "deployment": os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME"),
            "chunk_size": os.getenv("EMBEDDING_MODEL_CHUNK_SIZE"),
        },
        "openai-llm": {
            "api_version": os.getenv("LLM_API_VERSION"),
            "azure_deployment": os.getenv("LLM_DEPLOYMENT_NAME"),
            "api_key": os.getenv("LLM_API_KEY"),
            "azure_endpoint": os.getenv("LLM_API_ENDPOINT"),
        }
    }