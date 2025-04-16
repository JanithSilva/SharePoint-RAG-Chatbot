from pinecone import ServerlessSpec, Pinecone
from langchain_openai import AzureOpenAIEmbeddings
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreService:
    def __init__(self, config: Dict[str, str]):
        self.pinecone = Pinecone(api_key=config["pinecone_api_key"])
        self.index_name = config["index_name"]
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=config["deployment"],
            openai_api_version=config["api_version"],
            azure_endpoint=config["azure_endpoint"],
            api_key=config["api_key"],
            chunk_size=config["chunk_size"] 
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
        # Initialize index on class instantiation
        self.index = self.initialize_index()

    def initialize_index(self):
        # First check if index exists
        index_names = [index["name"] for index in self.pinecone.list_indexes()]
        if self.index_name not in index_names:
            # Create index if it doesn't exist
            self.pinecone.create_index(
                name=self.index_name,
                dimension=self.config["dimension"],
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        # Return the index (will connect to existing one if it exists)
        return self.pinecone.Index(self.index_name)

    def upsert_documents(self, documents: List[Dict]):
        vectors = []

        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc["text"])
            for j, chunk in enumerate(chunks):
                embedding = self.embeddings.embed_query(chunk)
                vector_id = f"doc_{i}_chunk_{j}"
                vectors.append((vector_id, embedding, {"text": chunk}))
        
        self.index.upsert(vectors=vectors)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [match.metadata["text"] for match in results.matches]