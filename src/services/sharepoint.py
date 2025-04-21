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
                file_id = hashlib.md5(
                    f"{file.unique_id}-{file.time_last_modified}".encode()
                ).hexdigest()
                
                files[file_id] = {
                    "name": file_name,
                    "server_path": file.properties["ServerRelativeUrl"],
                }
        
        return files
    
    def download_and_extract_text(self, file_details: Dict[str, dict]) -> List[Dict]:
        """
        Download documents from SharePoint in memory, process them using Azure Document Intelligence.
        """
        ctx = self.connect()
        results = {}

        for file_id, details in file_details.items():
            file = ctx.web.get_file_by_server_relative_url(details["server_path"])
            ctx.load(file)
            ctx.execute_query()

            #Download file content to memory
            file_stream = io.BytesIO()
            file.download(file_stream).execute_query()
            file_stream.seek(0)

            #Process with Azure Document Intelligence
            poller = self.document_analysis_client.begin_analyze_document("prebuilt-layout", file_stream)
            result = poller.result()

            enriched_details = details.copy()
            enriched_details["text"] = result.content
            results[file_id] = enriched_details

        return results

