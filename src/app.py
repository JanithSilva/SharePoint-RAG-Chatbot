import uuid
from typing import Dict
from fastapi import FastAPI, Request
from langgraph_sdk import get_client
from starlette.responses import JSONResponse
import time
import threading
import logging
from src.services.sharepoint import SharePointService
from src.services.document_ingest import DocumentIngestionService
from src.services.vector_store import VectorStoreService
from src.services.graph_store import GraphStoreService
from src.services.file_tracker import FileTracker
from src.settings import load_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_sharepoint():
    config = load_config()
    tracker = FileTracker()
    sharepoint = SharePointService(config["sharepoint"])
    doc_ai = DocumentIngestionService(config["azure_doc_intel"])
    vector_store = VectorStoreService({ 
            **config["pinecone"],
            **config["openai-embedding"]
        })
    graph_store = GraphStoreService()
    
    logger.info("Starting SharePoint monitor...")
    while True:
        try:
            # Check for new files every 5 minutes
            time.sleep(300)
            
            logger.info("Checking SharePoint for new documents...")
            all_files = sharepoint.get_all_files("Documents")
            new_files = tracker.get_new_files(all_files)
            
            if new_files:
                logger.info(f"Found {len(new_files)} new documents to process")
                temp_files = sharepoint.download_files(new_files)
                processed_docs = doc_ai.process_documents(list(temp_files.values()))
                
                # Update vector and graph stores
                vector_store.upsert_documents(processed_docs)

                for i, (file_id, doc) in enumerate(zip(new_files.keys(), processed_docs)):
                    graph_store.process_and_store_document(
                    f"doc_{file_id}",
                    doc["text"]
                    )   
                
                graph_store.create_indices()
                # Mark files as processed
                tracker.mark_files_processed(set(new_files.keys()))
                logger.info("Processing complete for new documents")
            else:
                logger.info("No new documents found")
                
        except Exception as e:
            logger.info(f"Error monitoring SharePoint: {e}")

def initialize_system():
    logger.info("System initializing...")
    config = load_config()
    tracker = FileTracker()
    sharepoint = SharePointService(config["sharepoint"])
    
    # Get all existing files
    all_files = sharepoint.get_all_files("Documents")
    new_files = tracker.get_new_files(all_files)
    
    if new_files:
        logger.info(f"Found {len(new_files)} new documents to process...")
        doc_ai = DocumentIngestionService(config["azure_doc_intel"])
        vector_store = VectorStoreService({ 
            **config["pinecone"],
            **config["openai-embedding"]
        })
        graph_store = GraphStoreService()
        
        temp_files = sharepoint.download_files(new_files)
        processed_docs = doc_ai.process_documents(list(temp_files.values()))
        
        vector_store.upsert_documents(processed_docs)

        for i, (file_id, doc) in enumerate(zip(new_files.keys(), processed_docs)):
            graph_store.process_and_store_document(
            f"doc_{file_id}",
            doc["text"]
            )
        graph_store.create_indices()
        tracker.mark_files_processed(set(new_files.keys()))

    logger.info("Initialization complete!")


initialize_system()

# Start sharepoint monitoring in a separate thread
monitor_thread = threading.Thread(target=monitor_sharepoint, daemon=True)
monitor_thread.start()

# Initialize the LangGraph client
langgraph_client = get_client()

# Initialize FastAPI app
app = FastAPI()


def get_user_id(request: Request) -> str:
    return "89df4336-297e-482d-a440-0eb11752180b" # Placeholder for user_id for testing


@app.post("/conversations/")
async def create_conversation(request: Request) -> Dict[str, str]:
    """Create a new conversation thread."""
    user_id = get_user_id(request)
    thread_id = str(uuid.uuid4())
    await langgraph_client.threads.create(
        thread_id=thread_id, metadata={"user_id": user_id}
    )
    return {"thread_id": thread_id, "user_id": user_id}

@app.post("/conversations/{thread_id}/send-message")
async def send_message(request: Request, thread_id: str):
    """Send a message to the assistant and get the response."""
    try:
        form_data = await request.json()
        msg = form_data.get("msg", "")

        if not msg or msg.isspace():
            return JSONResponse(status_code=400, content={"error": "Message cannot be empty"})

        run = await langgraph_client.runs.wait(
            thread_id=thread_id,
            assistant_id="agent",
            input={"question": msg, "max_retries": 3},
        )


        return {"human":run["question"], "ai": run["generation"]["content"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/conversations/")
async def list_conversations(request: Request):
    """List all conversation threads for the user."""
    user_id = get_user_id(request)
    threads = await langgraph_client.threads.search(
        metadata={"user_id": user_id}, limit=50, offset=0
    )
    return threads


@app.get("/conversations/{thread_id}")
async def get_conversation(thread_id: str):
    """Get the state of a specific conversation."""
   
    state = await langgraph_client.threads.get_state(thread_id)
    # conversation = []
    # for message in state["values"]["messages"]:
    #     if message["type"] in ("human", "ai"):
    #         # Skip AI messages that are just tool calls (empty content)
    #         if message["type"] == "ai" and not message["content"].strip():
    #             continue
    #         conversation.append({
    #             "role": message["type"],  # "human" or "ai"
    #             "content": message["content"]
    #         })

    return state
    


