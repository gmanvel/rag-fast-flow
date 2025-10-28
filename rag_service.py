"""
RAG Service for retrieving relevant context from Qdrant vector database.
"""

import os
import json
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser


class RAGService:
    """Service for RAG operations: embedding generation and vector search."""

    def __init__(
        self,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        ollama_base_url: Optional[str] = None,
        collection_name: str = "fast_flow",
        top_k: int = 3
    ):
        """
        Initialize RAG service.

        Args:
            qdrant_host: Qdrant host (default: localhost)
            qdrant_port: Qdrant port (default: 6333)
            ollama_base_url: Ollama base URL for embeddings (default: http://host.docker.internal:11434)
            collection_name: Qdrant collection name (default: fast_flow)
            top_k: Number of top results to retrieve (default: 3)
        """
        self.qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "host.docker.internal")
        self.qdrant_port = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))
        self.ollama_base_url = ollama_base_url or os.getenv(
            "OLLAMA_BASE_URL",
            "http://host.docker.internal:11434"
        )
        self.collection_name = collection_name
        self.top_k = top_k

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=self.qdrant_host,
            port=self.qdrant_port
        )

        # Initialize embedding model
        self.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url=self.ollama_base_url
        )

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant context for a query from the vector database.

        Args:
            query: User's question

        Returns:
            Formatted context string from top matching sections
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embed_model.get_text_embedding(query)

            # Search Qdrant for similar sections
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=self.top_k,
                with_payload=True
            )

            # Format the retrieved context
            if not search_results.points:
                return "No relevant context found in the knowledge base."

            context_parts = []
            for i, point in enumerate(search_results.points, 1):
                title = point.payload.get("title", "Untitled")
                text = point.payload.get("text", "")
                score = point.score

                context_parts.append(
                    f"[Section {i} - Relevance: {score:.2f}]\n"
                    f"Title: {title}\n"
                    f"Content: {text}\n"
                )

            return "\n".join(context_parts)

        except Exception as e:
            return f"Error retrieving context: {str(e)}"

    def check_connection(self) -> Dict[str, any]:
        """
        Check connection to Qdrant and verify collection exists.

        Returns:
            Dictionary with connection status and collection info
        """
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name in collection_names:
                collection_info = self.qdrant_client.get_collection(
                    collection_name=self.collection_name
                )
                return {
                    "status": "connected",
                    "collection_exists": True,
                    "points_count": collection_info.points_count,
                    "vector_size": collection_info.config.params.vectors.size
                }
            else:
                return {
                    "status": "connected",
                    "collection_exists": False,
                    "available_collections": collection_names
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def populate_database(self, json_file_path: str) -> Dict[str, any]:
        """
        Populate Qdrant database with embeddings from fast_flow_extracted.json.
        Uses semantic chunking to split sections into smaller, coherent pieces.

        Args:
            json_file_path: Path to fast_flow_extracted.json file

        Returns:
            Dictionary with status, sections_processed, and chunks_created
        """
        try:
            # Step 1: Delete existing collection if it exists
            try:
                self.qdrant_client.delete_collection(
                    collection_name=self.collection_name
                )
            except Exception:
                pass  # Collection doesn't exist, that's fine

            # Step 2: Create new collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

            # Step 3: Load JSON data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Step 4: Extract and filter sections
            sections_data = []
            for header in data:
                for section in header["sections"]:
                    title = section.get("tile", "").strip()
                    content = section.get("content", "").strip()
                    # Filter: non-empty title AND title != "Summary" AND has content
                    if title and title != "Summary" and content:
                        sections_data.append({"title": title, "text": content})

            if not sections_data:
                return {
                    "success": False,
                    "error": "No valid sections found in JSON file"
                }

            # Step 5: Initialize semantic splitter
            ollama_embeddings = OllamaEmbedding(
                model_name="nomic-embed-text",
                base_url=self.ollama_base_url
            )
            splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=70,
                embed_model=ollama_embeddings
            )

            # Step 6: Process each section into semantic chunks
            points = []
            for index, section in enumerate(sections_data):
                # Split section content into semantic nodes/chunks
                nodes = splitter.get_nodes_from_documents(
                    documents=[Document(text=section["text"])]
                )
                chunks = [(node.embedding, node.get_content()) for node in nodes]

                # Create point for each chunk
                for inner_index, (_, content) in enumerate(chunks):
                    # Skip empty chunks or chunks that are exactly "Summary"
                    if not content.strip() or content.strip() == "Summary":
                        continue

                    # Generate embedding for chunk
                    emb = ollama_embeddings.get_text_embedding(content)

                    # Create point with parent section's title
                    point = PointStruct(
                        id=index * 10 + inner_index,
                        vector=emb,
                        payload={
                            "title": section["title"],
                            "text": content  # Chunk content, not original section
                        }
                    )
                    points.append(point)

            if not points:
                return {
                    "success": False,
                    "error": "No valid chunks created from sections"
                }

            # Step 7: Batch insert all points into Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            return {
                "success": True,
                "sections_processed": len(sections_data),
                "chunks_created": len(points)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
