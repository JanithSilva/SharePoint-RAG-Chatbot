from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.authentication_context import AuthenticationContext
from typing import Dict
import hashlib
import tempfile

class SharePointService:
    def __init__(self, config: dict):
        self.config = config
        
    def connect(self):
        auth_ctx = AuthenticationContext(self.config["url"])
        auth_ctx.acquire_token_for_user(
            self.config["username"],
            self.config["password"]
        )
        return ClientContext(self.config["url"], auth_ctx)
    
    def get_all_files(self, library_name: str) -> Dict[str, dict]:
        """Return dict of {file_id: file_details}"""
        ctx = self.connect()
        lib = ctx.web.lists.get_by_title(library_name)
        items = lib.items.get().execute_query()
        
        files = {}
        for item in items:
            file = item.file
            ctx.load(file)
            ctx.execute_query()
            
            # Create unique ID based on file properties
            file_id = hashlib.md5(
                f"{file.unique_id}-{file.time_last_modified}".encode()
            ).hexdigest()
            
            files[file_id] = {
                "name": file.properties["Name"],
                "server_path": file.properties["ServerRelativeUrl"],
                "last_modified": file.properties["TimeLastModified"],
                "size": file.properties["Length"]
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