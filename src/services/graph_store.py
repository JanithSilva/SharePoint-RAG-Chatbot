from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from typing import List, Dict
from src.settings import load_config

class GraphStoreService:
    def __init__(self):
        config = load_config()

        neo4j_config = config["neo4j"]
        self.neo4j_graph = Neo4jGraph(
            url=neo4j_config["uri"],
            username=neo4j_config["user"],
            password=neo4j_config["password"]
        )
        
        openai_config = config["openai-llm"]
        self.llm = AzureChatOpenAI(
            azure_deployment=openai_config["azure_deployment"],
            openai_api_version=openai_config["api_version"],
            azure_endpoint=openai_config["azure_endpoint"],
            api_key=openai_config["api_key"]
        )
        
        self.graph_transformer = LLMGraphTransformer(llm=self.llm)
        
        self.text_splitter = TokenTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )

        embedding_config = config["openai-embedding"]
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embedding_config["deployment"],
            openai_api_version=embedding_config["api_version"],
            azure_endpoint=embedding_config["azure_endpoint"],
            api_key=embedding_config["api_key"],
            chunk_size=embedding_config["chunk_size"]
        )

    def _create_document_chunks(self, text: str) -> List[Document]:
        """Split document into manageable chunks for graph processing"""
        chunks = self.text_splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]

    def process_and_store_document(self, doc_id: str, text: str) -> Dict:
        """
        Process document text into knowledge graph components and store in Neo4j
        Returns statistics about the processed graph
        """
        # Step 1: Split document into chunks
        document_chunks = self._create_document_chunks(text)
        
        # Step 2: Transform each chunk into graph components
        graph_documents = []
        for chunk in document_chunks:
            graph_documents.extend(
                self.graph_transformer.convert_to_graph_documents([chunk])
            )
        
        # Step 3: Store in Neo4j
        self.neo4j_graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        
        # Step 4: Create document node and connect to extracted entities
        self._create_document_node(doc_id, text, graph_documents)
        
        return {
            "nodes_created": len(graph_documents),
            "relationships_created": sum(len(doc.relationships) for doc in graph_documents),
            "chunks_processed": len(document_chunks)
        }

    def _create_document_node(self, doc_id: str, text: str, graph_docs: List):
        """Create central document node and connect to extracted entities"""
        # Create document node
        self.neo4j_graph.query("""
            MERGE (d:Document {id: $doc_id})
            SET d.text = $text,
                d.summary = $summary,
                d.processed_at = datetime()
        """, {"doc_id": doc_id, "text": text, "summary": text[:200] + "..."})
        
        # Connect to all extracted entities
        for doc in graph_docs:
            for node in doc.nodes:
                self.neo4j_graph.query("""
                    MATCH (d:Document {id: $doc_id})
                    MERGE (e:Entity {id: $entity_id})
                    SET e += $properties
                    MERGE (d)-[:CONTAINS_ENTITY]->(e)
                """, {
                    "doc_id": doc_id,
                    "entity_id": node.id,
                    "properties": node.properties
                })

    def query_entities(self, query: str, limit: int = 5) -> List[Dict]:
        """Enhanced entity query with semantic search"""
        result = self.neo4j_graph.query("""
            CALL db.index.vector.queryNodes(
                'entity_embeddings', 
                $limit, 
                $query_embedding
            ) YIELD node, score
            RETURN node.id as entity_id, 
                   node.type as entity_type,
                   node.description as description,
                   score
            ORDER BY score DESC
            LIMIT $limit
        """, {
            "query_embedding": self._get_text_embedding(query),
            "limit": limit
        })
        return result

    def get_document_entities(self, doc_id: str) -> List[Dict]:
        """Get all entities for a specific document"""
        result = self.neo4j_graph.query("""
            MATCH (d:Document {id: $doc_id})-[:CONTAINS_ENTITY]->(e)
            RETURN e.id as entity_id, 
                   e.type as entity_type,
                   e.description as description
            ORDER BY e.type
        """, {"doc_id": doc_id})
        return result

    def _get_text_embedding(self, text: str) -> List[float]:
        """Helper method to get text embeddings (simplified)"""
    
        return self.embeddings.embed_query(text)

    def create_indices(self):
        """Create necessary database indices"""
        self.neo4j_graph.query("""
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (e:Entity)
            ON e.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
        """)
        
        self.neo4j_graph.query("""
            CREATE INDEX document_index IF NOT EXISTS
            FOR (d:Document)
            ON (d.id)
        """)

    def clear_graph(self):
        """Clear all graph data (for testing)"""
        self.neo4j_graph.query("MATCH (n) DETACH DELETE n")