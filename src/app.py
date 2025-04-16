from fastapi import FastAPI
from langgraph_sdk import get_client
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
    logger.info("Started SharePoint monitor thread")
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
        logger.info("Checking SharePoint for new documents...")
        all_files = sharepoint.get_all_files()
        new_files = tracker.get_new_files(all_files)
        
        if new_files:
            logger.info(f"Found {len(new_files)} new documents to process")
            temp_files = sharepoint.download_files(new_files)
            processed_docs = doc_ai.process_documents(list(temp_files.values()))
            vector_store.upsert_documents(processed_docs)
            # Process and store documents in graph store 
            for i, (file_id, doc) in enumerate(zip(new_files.keys(), processed_docs)):
                
                result = graph_store.process_and_store_document(
                    text=doc["text"],
                    doc_id=file_id
                    )   
            tracker.mark_files_processed(set(new_files.keys()))
            logger.info("Processing complete for new documents")
        else:
            logger.info("No new documents found")
        # Check for new files every 5 minutes
        time.sleep(300)    
       

monitor_thread = threading.Thread(target=monitor_sharepoint, daemon=True)
monitor_thread.start()

langgraph_client = get_client()

# Initialize FastAPI app
app = FastAPI()



    


