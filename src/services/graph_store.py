from typing import List, Dict, Optional, Union
import logging
from datetime import datetime
from langchain_community.graphs.graph_document import GraphDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from src.settings import load_config
import json



# Configure logging
logger = logging.getLogger(__name__)

class GraphStoreService:
    """Service for processing text documents into knowledge graphs stored in Neo4j."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the GraphStoreService with configurations for Neo4j, LLM, and embeddings.
        
        Args:
            config_path: Optional path to the configuration file
        """
        try:
            config = load_config()

            # Initialize Neo4j connection
            neo4j_config = config["neo4j"]
            self.neo4j_graph = Neo4jGraph(
                url=neo4j_config["uri"],
                username=neo4j_config["user"],
                password=neo4j_config["password"]
            )
            
            # Initialize LLM
            openai_config = config["openai-llm"]
            self.llm = AzureChatOpenAI(
                azure_deployment=openai_config["azure_deployment"],
                openai_api_version=openai_config["api_version"],
                azure_endpoint=openai_config["azure_endpoint"],
                api_key=openai_config["api_key"]
            )
            
            self.graph_transformer = LLMGraphTransformer(llm=self.llm)
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            # Initialize embeddings
            embedding_config = config["openai-embedding"]
            self.embedding_dimension = embedding_config.get("dimension", 1536)
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=embedding_config["deployment"],
                openai_api_version=embedding_config["api_version"],
                azure_endpoint=embedding_config["azure_endpoint"],
                api_key=embedding_config["api_key"],
                chunk_size=embedding_config["chunk_size"]
            )
            
            # Ensure indices are created
            self.create_indices()
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphStoreService: {str(e)}")
            raise

    def _create_document_chunks(self, text: str) -> List[Document]:
        """
        Split document into manageable chunks for graph processing.
        
        Args:
            text: The document text to split
            
        Returns:
            List of Document objects
        """
        chunks = self.text_splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]

    def process_and_store_document(self, text: str, doc_id: str) -> Dict:
        """
        Process document text into knowledge graph components and store in Neo4j.
        
        Args:
            text: The document text to process
            doc_id: Unique identifier for the document
            
        Returns:
            Dictionary with statistics about the processed graph
        """
        try:
            # Step 1: Split document into chunks
            document_chunks = self._create_document_chunks(text)
            logger.info(f"Created {len(document_chunks)} chunks from document {doc_id}")
            
            # Step 2: Transform chunks into graph components (batched for efficiency)
            graph_documents = []
            batch_size = 5 
            for i in range(0, len(document_chunks), batch_size):
                batch = document_chunks[i:i+batch_size]
                
                # Generate embeddings for the batch
                chunk_texts = [chunk.page_content for chunk in batch]
                chunk_embeddings = self.embeddings.embed_documents(chunk_texts)
                
                # Convert to graph documents
                graph_batch = self.graph_transformer.convert_to_graph_documents(batch)
                
                for doc, embedding in zip(graph_batch, chunk_embeddings):
                    doc.source.metadata["embedding"] = embedding

                graph_documents.extend(graph_batch)
                logger.debug(f"Processed batch {i//batch_size + 1} of {len(document_chunks)//batch_size + 1}")
            
            # Step 3: Store in Neo4j (basic structure)
            self.neo4j_graph.add_graph_documents(
                graph_documents,
                include_source=True,
                baseEntityLabel=True  # This ensures the source document is linked
            )
        
            
            result = {
                "nodes_created": sum(len(doc.nodes) for doc in graph_documents),
                "relationships_created": sum(len(doc.relationships) for doc in graph_documents),
                "chunks_processed": len(document_chunks),
                "document_id": doc_id
            }
            logger.info(f"Successfully processed document {doc_id}: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            raise

    
   

    def create_indices(self):
        """
        Create a vector index in Neo4j for the embeddings if it doesn't exist.
        """
        index_query = """
        CREATE VECTOR INDEX `document_embeddings` IF NOT EXISTS
        FOR (n:Document)
        ON n.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: $dimension,
            `vector.similarity_function`: 'cosine'
        }}
        """
        try:
            self.neo4j_graph.query(
                index_query,
                params={"dimension": self.embedding_dimension}
            )
            logger.info("Created vector index for chunk embeddings")
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
            raise


    def query_semantically(self, question: str, top_k: int = 5, score_threshold: float = 0.75) -> str:
        """
        Query the knowledge graph using semantic similarity to find relevant entities and their relationships.
        
        Args:
            question: The natural language question to query with
            top_k: Number of most similar chunks to return
            score_threshold: Minimum similarity score to include results
            
        Returns:
            List of dictionaries containing relevant entities, their relationships, and connected nodes
        """
       
        #Embed the question
        question_embedding = self.embeddings.embed_query(question)
        
        #Query the vector index for similar chunks
        vector_query = """
        CALL db.index.vector.queryNodes(
            'document_embeddings', 
            $top_k, 
            $question_embedding
        ) YIELD node, score
        WHERE score >= $score_threshold
        RETURN node, score
        ORDER BY score DESC
        """
        vector_results = self.neo4j_graph.query(
            vector_query,
            params={
                "top_k": top_k,
                "question_embedding": question_embedding,
                "score_threshold": score_threshold
            }
        )
    
        if not vector_results:
            logger.info(f"No semantically similar chunks found for question: {question}")
            return []
        
        # Get related entities and their relationships for each matching chunk
        results = []
        for record in vector_results:
            chunk_node = record["node"]
            
            entity_query = """
            MATCH (chunk:Document {id: $chunk_id})
            OPTIONAL MATCH (chunk)-[r1]-(entity)
            WHERE NOT entity:Document  // Exclude other document chunks
            // Now find all relationships for these entities
            OPTIONAL MATCH (entity)-[r2]-(related_node)
            WHERE NOT related_node:Document  // Exclude document nodes
            RETURN 
                entity, 
                labels(entity) as entity_labels, 
                collect(DISTINCT {
                    relationship: r2,
                    type: type(r2),
                    direction: CASE WHEN startNode(r2) = entity THEN 'FORWARD' ELSE 'BACKWARD' END,
                    related_node: related_node,
                    related_node_labels: labels(related_node)
                }) as entity_relationships
            """
            entities = self.neo4j_graph.query(
                entity_query,
                params={"chunk_id": chunk_node["id"]}
            )
            results.append(entities)
   
        formatted_output = [] 

        #Process each entity to build a structured output
        for entity_data in results:
            if not entity_data:
                continue
                
            for entity_entry in entity_data:
                if not isinstance(entity_entry, dict):
                    continue
                entity = entity_entry.get('entity', {}).get('id', 'Unknown')
                labels = ', '.join(entity_entry.get('entity_labels', []))
                
                #Start building the entity description
                entity_desc = f"Entity: {entity} (Labels: {labels})\n"
                
                #Process relationships
                relationships = entity_entry.get('entity_relationships', [])
                if relationships:
                    entity_desc += "Relationships:\n"
                    for rel in relationships:
                        if rel["relationship"] is not None:
                            rel_type = rel.get('type', '')
                            direction = rel.get('direction', '')
                            related_node = rel.get('related_node', {}).get('id', 'Unknown')
                            related_labels = ', '.join(rel.get('related_node_labels', []))
                            if direction == 'FORWARD':
                                rel_desc = f"  - {entity} -> {rel_type} -> {related_node} (Labels: {related_labels})"
                            else: 
                                rel_desc = f"  - {related_node} -> {rel_type} -> {entity} (Labels: {related_labels})"
                            
                            entity_desc += rel_desc + "\n"
                        else:
                            continue
                
                formatted_output.append(entity_desc)
        
        #Join all entity descriptions with separators
        return "\n\n".join(formatted_output)
      
    

    