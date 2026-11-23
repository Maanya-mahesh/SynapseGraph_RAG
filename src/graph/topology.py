#Step - 3: Graph Topology & Hierarchical Leiden Community Detection
from dataclasses import dataclass, field
import logging
from typing import Dict, List, Tuple, Any
import networkx as nx
from cdlib import algorithms, NodeClustering

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class CommunityNode:#Represents a hierarchical community cluster extracted via Leiden CPM
    community_id: int
    level: int
    members: List[str]
    density: float
    hub_entities: List[Tuple[str, float]]

class KnowledgeGraphTopologyEngine: #Engine for building directed Knowledge Graphs and extracting Leiden communities
    def __init__(self, resolution_parameter: float = 0.05) -&gt; None:
        self.graph = nx.DiGraph()
        self.resolution = resolution_parameter
    def build_graph_from_triples(self, triples: List[Any]) -&gt; None: #Constructs a directed NetworkX graph from a list of SPOTriple objects
        for t in triples:
            # Handle both dataclass/Pydantic object and tuple representations
            subject = getattr(t, "subject", t[0] if isinstance(t, (list, tuple)) else "")
            predicate = getattr(t, "predicate", t[1] if isinstance(t, (list, tuple)) else "")
            obj = getattr(t, "object", t[2] if isinstance(t, (list, tuple)) else "")
            confidence = getattr(t, "confidence", 1.0)
            if not subject or not obj:
                continue
            # Add nodes
            self.graph.add_node(subject)
            self.graph.add_node(obj)
            # Add or update directed edge
            if self.graph.has_edge(subject, obj):
                self.graph[subject][obj]["weight"] += confidence
            else:
                self.graph.add_edge(subject, obj, relation=predicate, weight=confidence)
        logger.info(
            f"Built Knowledge Graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges."
        )
    def compute_degree_centrality(self, top_k: int = 10) -&gt; List[Tuple[str, float]]:#Computes Normalized Degree Centrality to identify top Hub Entities
        if self.graph.number_of_nodes() == 0:
            return []
        centrality = nx.degree_centrality(self.graph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]
    def detect_leiden_communities(self, level: int = 0) -&gt; List[CommunityNode]:#Executes the Leiden algorithm using Constant Potts Model (CPM) quality optimization
        if self.graph.number_of_nodes() &lt; 2:
            logger.warning("Graph contains fewer than 2 nodes. Community detection skipped.")
            return []
        # Convert directed graph to undirected for community clustering
        undirected_graph = self.graph.to_undirected()
        # Execute Leiden CPM via cdlib
        clustering: NodeClustering = algorithms.leiden(
            undirected_graph, 
            initial_membership=None, 
            weights=None
        )

        community_results = []
        centrality_map = nx.degree_centrality(self.graph)

        for cid, community_members in enumerate(clustering.communities):
            subgraph = undirected_graph.subgraph(community_members)
            n_c = len(community_members)
            e_c = subgraph.number_of_edges()
            max_possible_edges = (n_c * (n_c - 1)) / 2.0 if n_c &gt; 1 else 1.0
            density = round(e_c / max_possible_edges, 4)

            # Extract hub entities within this specific community
            hubs = sorted(
                [(node, round(centrality_map[node], 4)) for node in community_members],
                key=lambda x: x[1],
                reverse=True
            )[:3]

            community_results.append(
                CommunityNode(
                    community_id=cid,
                    level=level,
                    members=community_members,
                    density=density,
                    hub_entities=hubs
                )
            )

        logger.info(f"Extracted {len(community_results)} Leiden communities at Level {level}.")
        return community_results
