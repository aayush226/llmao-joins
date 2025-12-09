# file: llmao_joins/graph_and_clusters.py
from dataclasses import dataclass
from typing import Dict, Tuple

from neo4j import GraphDatabase
import networkx as nx

from .candidate_generation import CandidatePair
from .config import PipelineConfig

@dataclass
class GraphStats:
    n_nodes: int = 0
    n_edges: int = 0
    n_clusters: int = 0

class Neo4jGraph:
    """
    Simple wrapper around Neo4j driver for our value-alias graph.
    Nodes: (:Value {value: <normalized_string>, cluster_id?})
    Edges: (:Value)-[:ALIAS_WITH {rule_score, string_sim, embed_sim, llm_score, combined_score, sources}]-(:Value)
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def init_schema(self) -> None:
        """
        Ensure uniqueness on Value.value.
        """
        pass

    def upsert_pair(self, pair: CandidatePair) -> None:
        """
        Create or update nodes/edge for a candidate pair.
        """
        pass

    @staticmethod
    def _upsert_pair_tx(tx, pair: CandidatePair):
        cypher = """
        MERGE (a:Value {value: $left})
        MERGE (b:Value {value: $right})
        MERGE (a)-[r:ALIAS_WITH]->(b)
        SET r.rule_score = $rule_score,
            r.string_sim = $string_sim,
            r.embed_sim = $embed_sim,
            r.llm_score = $llm_score,
            r.combined_score = $combined_score,
            r.sources = $sources
        """
        tx.run(
            cypher,
            left=pair.left_norm,
            right=pair.right_norm,
            rule_score=pair.rule_score,
            string_sim=pair.string_sim,
            embed_sim=pair.embed_sim,
            llm_score=pair.llm_score,
            combined_score=pair.combined_score,
            sources=pair.sources,
        )

    def export_graph_for_threshold(self, threshold: float) -> nx.Graph:
        """
        Export an undirected graph of Value nodes for edges with combined_score >= threshold.
        """
        return nx.Graph()

    def write_cluster_ids(self, cluster_map: Dict[str, int]) -> None:
        """
        Write cluster_id back onto nodes for explainability / re-use.
        """
        pass

def build_graph_and_clusters(
    pairs: Dict[Tuple[str, str], CandidatePair],
    cfg: PipelineConfig,
) -> GraphStats:
    """
    Build the value-alias graph in Neo4j and compute synonym clusters via
    connected components on high-confidence edges.
    """
    graph_stats = GraphStats()
    neo = Neo4jGraph(cfg)
    neo.init_schema()

    # Write all pairs to Neo4j with their scores
    for pair in pairs.values():
        neo.upsert_pair(pair)

    # Export accepted subgraph (combined_score >= accept_threshold)
    G = neo.export_graph_for_threshold(cfg.accept_threshold)

    # Connected components => synonym sets
    cluster_map: Dict[str, int] = {}
    for cluster_id, component in enumerate(nx.connected_components(G), start=1):
        for value in component:
            cluster_map[value] = cluster_id

    neo.write_cluster_ids(cluster_map)
    neo.close()

    graph_stats.n_nodes = G.number_of_nodes()
    graph_stats.n_edges = G.number_of_edges()
    graph_stats.n_clusters = len({cid for cid in cluster_map.values()})
    return graph_stats
