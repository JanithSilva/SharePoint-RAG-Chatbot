from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.authentication_context import AuthenticationContext
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict
import hashlib
import tempfile
import json
import os
import io

class SharePointService:
    def __init__(self, config: dict):
        self.config = config

        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=config["endpoint"],
            credential=AzureKeyCredential(config["key"])
        )
        
    def connect(self):
        auth_ctx = AuthenticationContext(self.config["url"])
        auth_ctx.acquire_token_for_user(
            self.config["username"],
            self.config["password"]
        )
        return ClientContext(self.config["url"], auth_ctx)
    
    def get_all_files(self) -> Dict[str, dict]:
        """Return dict of {file_id: file_details} for text documents only"""
        ctx = self.connect()
        lib = ctx.web.lists.get_by_title(self.config["library_name"])
        items = lib.items.get().execute_query()
        
        allowed_extensions = {".txt", ".doc", ".docx", ".pdf"}
        files = {}
        
        for item in items:
            file = item.file
            ctx.load(file)
            ctx.execute_query()
            
            file_name = file.properties["Name"]
            _, ext = os.path.splitext(file_name.lower())
            
            if ext in allowed_extensions:
                # Unique ID based on file properties
                file_id = hashlib.md5(
                    f"{file.unique_id}-{file.time_last_modified}".encode()
                ).hexdigest()
                
                files[file_id] = {
                    "name": file_name,
                    "server_path": file.properties["ServerRelativeUrl"],
                }
        
        return files
    
    def download_files(self, file_details: Dict[str, dict]) -> Dict[str, str]:
        """Download files and return {file_id: temp_file_path}"""
        ctx = self.connect()
        temp_files = {}
        
        for file_id, details in file_details.items():
            file = ctx.web.get_file_by_server_relative_url(details["server_path"])
            ctx.load(file)
            ctx.execute_query()
            
            _, temp_path = tempfile.mkstemp()
            with open(temp_path, "wb") as local_file:
                file.download(local_file).execute_query()
            temp_files[file_id] = temp_path
            
        return temp_files
    
    def process_documents(self, file_paths: List[str]) -> List[Dict]:
        results = []
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                poller = self.client.begin_analyze_document("prebuilt-layout", f)
                result = poller.result()
                results.append( result.content)
        return results
    

    def download_and_extract_text(self, file_details: Dict[str, dict]) -> List[Dict]:
        """
        Download documents from SharePoint in memory, process them using Azure Document Intelligence,
        and return a list of extracted content.
        """
        ctx = self.connect()
        results = {}

        for file_id, details in file_details.items():
            file = ctx.web.get_file_by_server_relative_url(details["server_path"])
            ctx.load(file)
            ctx.execute_query()

            # Download file content to memory
            file_stream = io.BytesIO()
            file.download(file_stream).execute_query()
            file_stream.seek(0)

            # Process the in-memory file with Azure Document Intelligence
            poller = self.document_analysis_client.begin_analyze_document("prebuilt-layout", file_stream)
            result = poller.result()

            enriched_details = details.copy()
            enriched_details["text"] = result.content
            results[file_id] = enriched_details

        return results


def get_metadata(self, library_name: str) -> str:
    """
    Return all metadata fields for files in a specific SharePoint library
    as a structured string suitable for LLM context (in JSON Lines format).
    """
    ctx = self.connect()
    lib = ctx.web.lists.get_by_title(library_name)
    items = lib.items.get().execute_query()

    metadata_lines = []

    for item in items:
        file = item.file
        ctx.load(file)
        ctx.execute_query()

        # Safely convert all metadata to a regular dictionary (if not already)
        metadata_dict = dict(file.properties)

        # Optionally clean up nested objects or non-serializable data
        for key, value in metadata_dict.items():
            if hasattr(value, 'properties'):
                metadata_dict[key] = dict(value.properties)
            elif isinstance(value, bytes):
                metadata_dict[key] = str(value)
            elif isinstance(value, (int, float, str, list, dict, type(None))):
                continue
            else:
                metadata_dict[key] = str(value)  # Fallback to string

        metadata_lines.append(json.dumps(metadata_dict, ensure_ascii=False))

    return "\n".join(metadata_lines)