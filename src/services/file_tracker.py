import json
import os
from typing import Set, Dict
from pathlib import Path

class FileTracker:
    def __init__(self, tracking_file: str = "data/processed_files.json"):
        self.tracking_file = tracking_file
        Path("data").mkdir(exist_ok=True)
        if not os.path.exists(self.tracking_file):
            with open(self.tracking_file, "w") as f:
                json.dump({"processed_files": []}, f)
    
    def load_processed_files(self) -> Set[str]:
        with open(self.tracking_file, "r") as f:
            data = json.load(f)
        return set(data.get("processed_files", []))
    
    def save_processed_files(self, file_ids: Set[str]):
        with open(self.tracking_file, "w") as f:
            json.dump({"processed_files": list(file_ids)}, f)
    
    def get_new_files(self, current_files: Dict[str, str]) -> Dict[str, str]:
        """Return only files that haven't been processed before"""
        processed = self.load_processed_files()
        return {file_id: details 
                for file_id, details in current_files.items()
                if file_id not in processed}
    
    def mark_files_processed(self, file_ids: Set[str]):
        processed = self.load_processed_files()
        processed.update(file_ids)
        self.save_processed_files(processed)