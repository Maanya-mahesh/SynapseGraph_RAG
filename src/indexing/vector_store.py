import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LocalVectorStore:# Manages local persistent ChromaDB vector storage with dual-collection indexing
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -&gt; None:
        self.persist_directory = persist_directory
        self.model_name = model_name

        logger.info(f"Initializing embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)

        logger.info(f"Initializing persistent ChromaDB client at: {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Initialize dual collections
        self.prose_collection = self.client.get_or_create_collection(
            name="prose_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.summary_collection = self.client.get_or_create_collection(
            name="community_summaries",
            metadata={"hnsw:space": "cosine"}
        )

    def _generate_embeddings(self, texts: List[str]) -&gt; List[List[float]]: #Generates dense vector embeddings using sentence-transformers
        if not texts:
            return []
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def add_prose_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -&gt; None: #Indexes layout-aware prose text chunks into the 'prose_chunks' collections
        if not texts or len(texts) != len(chunk_ids) or len(texts) != len(metadatas):
            raise ValueError("Mismatched lengths among chunk_ids, texts, and metadatas.")

        logger.info(f"Embedding and indexing {len(texts)} prose chunks...")
        embeddings = self._generate_embeddings(texts)

        self.prose_collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Successfully indexed {len(texts)} prose chunks into 'prose_chunks'.")

    def add_community_summaries(
        self,
        summary_ids: List[str],
        summaries: List[str],
        metadatas: List[Dict[str, Any]]
    ) -&gt; None: #Indexes Leiden community summaries into the 'community_summaries' collection
        if not summaries or len(summaries) != len(summary_ids) or len(summaries) != len(metadatas):
            raise ValueError("Mismatched lengths among summary_ids, summaries, and metadatas.")

        logger.info(f"Embedding and indexing {len(summaries)} community summaries...")
        embeddings = self._generate_embeddings(summaries)

        self.summary_collection.upsert(
            ids=summary_ids,
            embeddings=embeddings,
            documents=summaries,
            metadatas=metadatas
        )
        logger.info(f"Successfully indexed {len(summaries)} community summaries into 'community_summaries'.")

    def query_prose_chunks(
        self,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -&gt; Dict[str, Any]: #Executes semantic similarity search over prose_chunks with optional metadata filtering
        query_embedding = self._generate_embeddings([query_text])[0]
        
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        return self.prose_collection.query(**query_kwargs)

    def query_community_summaries(
        self,
        query_text: str,
        n_results: int = 3,
        where_filter: Optional[Dict[str, Any]] = None
    ) -&gt; Dict[str, Any]: #Executes semantic similarity search over community_summaries with optional metadata filtering
        query_embedding = self._generate_embeddings([query_text])[0]

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        return self.summary_collection.query(**query_kwargs)
