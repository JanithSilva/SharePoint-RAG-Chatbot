from fastapi import FastAPI
from langgraph_sdk import get_client
import time
import threading
import logging
from src.services.sharepoint import SharePointService
from src.services.vector_store import VectorStoreService
from src.services.graph_store import GraphStoreService
from src.services.file_tracker import FileTracker
from src.settings import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_sharepoint():

    config = load_config()
    tracker = FileTracker()
    sharepoint = SharePointService({**config["sharepoint"], **config["azure_doc_intel"]})
    vector_store = VectorStoreService({ **config["pinecone"], **config["openai-embedding"]})
    graph_store = GraphStoreService()
    logger.info("Starting SharePoint monitor...")
    while True:
        try:
            logger.info("Checking SharePoint for new documents...")
            all_files = sharepoint.get_all_files()
            new_files = tracker.get_new_files(all_files)
            
            if new_files:
                logger.info(f"Found {len(new_files)} new documents to process")
                docs = sharepoint.download_and_extract_text(new_files)
                vector_store.upsert_documents(docs)
                # Process and store documents in graph store 
                for file_id, details in docs.items():
                    result = graph_store.process_and_store_document(
                        text=details["text"],
                        )
                    logger.info(f"Processed {details['name']} into graph with {result['nodes_created']} nodes and {result['relationships_created']} relationships")
                tracker.mark_files_processed(set(docs.keys()))
                logger.info("Processing complete for new documents")
            else:
                logger.info("No new documents found")

        except Exception as e:
            logger.error(f"Error during SharePoint monitoring cycle: {e}")
            
        # Check for new files every 5 minutes
        time.sleep(300)    
       

monitor_thread = threading.Thread(target=monitor_sharepoint, daemon=True)
monitor_thread.start()

#langgraph_client = get_client()

# Initialize FastAPI app
app = FastAPI()



    


