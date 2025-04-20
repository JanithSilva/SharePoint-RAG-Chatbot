from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict

class DocumentIngestionService:
    def __init__(self, config: dict):
        self.client = DocumentAnalysisClient(
            endpoint=config["endpoint"],
            credential=AzureKeyCredential(config["key"])
        )
    
    def process_documents(self, file_paths: List[str]) -> List[Dict]:
        results = []
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                poller = self.client.begin_analyze_document("prebuilt-layout", f)
                result = poller.result()
                results.append( result.content)
        return results