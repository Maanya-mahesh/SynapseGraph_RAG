import logging
from typing import List, Dict, Any, Tuple, Optional
from src.graph.topology import KnowledgeGraphTopologyEngine
from src.indexing.vector_store import LocalVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DualRouteRRFRetriever:"""Executes parallel Dual-Route retrieval (Graph Traversal + Dense Vector Search) &
fuses candidate rankings using Reciprocal Rank Fusion (RRF) """
    def __init__(
        self,
        graph_engine: KnowledgeGraphTopologyEngine,
        vector_store: LocalVectorStore,
        rrf_k: int = 60
    ) -&gt; None:
        self.graph_engine = graph_engine
        self.vector_store = vector_store
        self.rrf_k = rrf_k

    def _execute_route_a_graph(self, query_text: str, top_k: int = 5) -&gt; List[Dict[str, Any]]: #Route A: Executes topological graph traversal and community summary retrieval
        results = []
        # 1 Fetch relevant community summaries from vector store summary collection
        summary_matches = self.vector_store.query_community_summaries(query_text, n_results=top_k)
        if summary_matches and "documents" in summary_matches and summary_matches["documents"]:
            docs = summary_matches["documents"][0]
            metas = summary_matches["metadatas"][0] if "metadatas" in summary_matches else [{}] * len(docs)
            ids = summary_matches["ids"][0] if "ids" in summary_matches else [f"summary_{i}" for i in range(len(docs))]

            for doc_id, doc_text, meta in zip(ids, docs, metas):
                results.append({
                    "id": doc_id,
                    "content": doc_text,
                    "metadata": meta,
                    "source_route": "Route_A_Graph"
                })

        return results

    def _execute_route_b_vector(self, query_text: str, top_k: int = 5) -&gt; List[Dict[str, Any]]: #Route B: Executes dense vector similarity search over prose_chunks
        
        prose_matches = self.vector_store.query_prose_chunks(query_text, n_results=top_k)
        if prose_matches and "documents" in prose_matches and prose_matches["documents"]:
            docs = prose_matches["documents"][0]
            metas = prose_matches["metadatas"][0] if "metadatas" in prose_matches else [{}] * len(docs)
            ids = prose_matches["ids"][0] if "ids" in prose_matches else [f"prose_{i}" for i in range(len(docs))]

            for doc_id, doc_text, meta in zip(ids, docs, metas):
                results.append({
                    "id": doc_id,
                    "content": doc_text,
                    "metadata": meta,
                    "source_route": "Route_B_Vector"
                })

        return results

    def retrieve_and_fuse(
        self,
        query_text: str,
        top_k_final: int = 5,
        route_top_k: int = 10
    ) -&gt; List[Dict[str, Any]]: #Executes Dual-Route retrieval and merges candidate lists via Reciprocal Rank Fusion
        logger.info(f"Executing Dual-Route Search for query: '{query_text}'")
        # Step 1: Parallel Route Execution
        route_a_list = self._execute_route_a_graph(query_text, top_k=route_top_k)
        route_b_list = self._execute_route_b_vector(query_text, top_k=route_top_k)
        # Step 2: Build candidate map and rank score tracker
        rrf_scores: Dict[str, float] = {}
        candidate_map: Dict[str, Dict[str, Any]] = {}
        # Process Route A ranks
        for rank_idx, item in enumerate(route_a_list, start=1):
            doc_id = item["id"]
            candidate_map[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank_idx))
        # Process Route B ranks
        for rank_idx, item in enumerate(route_b_list, start=1):
            doc_id = item["id"]
            candidate_map[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank_idx))
        # Step 3: Sort candidates by descending RRF Score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        # Step 4: Assemble top-K fused result set
        fused_results = []
        for doc_id, rrf_score in sorted_candidates[:top_k_final]:
            candidate = candidate_map[doc_id].copy()
            candidate["rrf_score"] = round(rrf_score, 6)
            fused_results.append(candidate)
        logger.info(f"RRF Fusion complete. Retrieved {len(fused_results)} top-ranked context items.")
        return fused_results
