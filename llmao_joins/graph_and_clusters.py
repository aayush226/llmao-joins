# file: llmao_joins/graph_and_clusters.py

from dataclasses import dataclass, field
from typing import Dict, Tuple, List

from neo4j import GraphDatabase
import networkx as nx

from .candidate_generation import CandidatePair
from .config import PipelineConfig


@dataclass
class GraphStats:
    n_nodes: int = 0
    n_edges: int = 0
    n_clusters: int = 0
    # Map normalized value -> cluster id (used by pipeline to make the join ... so pretty important)
    cluster_map: Dict[str, int] = field(default_factory=dict)


class Neo4jGraph:
    """
    A wrapper around Neo4j driver for value alias graph.

    Nodes: (:Value {value: <normalized_string>, cluster_id?, seen_count?, first_seen?, last_seen?})
    Edges: (:Value)-[:ALIAS_WITH {
              rule_score?,
              string_sim?,
              embed_sim?,
              llm_score?,
              combined_score?,
              sources?,
              seen_count?
           }]-(:Value)

    We treat the graph as undirected for clustering (even though the stored
    relationship is directed in Neo4j).
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
        Ensure uniqueness on Value.value and an index on cluster_id.
        """
        with self.driver.session() as session:
            # Unique constraint on normalized value string
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Value) "
                "REQUIRE v.value IS UNIQUE"
            )

            # Index on cluster_id for faster lookups and  analytics
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (v:Value) ON (v.cluster_id)"
            )

    # ---------- WRITE / UPSERT ----------

    def upsert_pair(self, pair: CandidatePair) -> None:
        """
        Create or update nodes/edge for a candidate pair.

        - Make sure both Value nodes exist (MERGE).
        - Update simple node stats (how often we've seen this value).
        - Create or update the ALIAS_WITH relationship with the scores
          from this run. If the edge already exists, we update scores
          and bump a seen_count counter.
        """
        # If we have literally no signal at all, skip writing.
        if (
            pair.rule_score is None
            and pair.string_sim is None
            and pair.embed_sim is None
            and pair.llm_score is None
            and pair.combined_score is None
        ):
            return

        with self.driver.session() as session:
            session.write_transaction(self._upsert_pair_tx, pair)

    @staticmethod
    def _upsert_pair_tx(tx, pair: CandidatePair):
        cypher = """
        MERGE (a:Value {value: $left})
        ON CREATE SET
            a.seen_count = 1,
            a.first_seen = timestamp(),
            a.last_seen = timestamp()
        ON MATCH SET
            a.seen_count = coalesce(a.seen_count, 0) + 1,
            a.last_seen = timestamp()

        MERGE (b:Value {value: $right})
        ON CREATE SET
            b.seen_count = 1,
            b.first_seen = timestamp(),
            b.last_seen = timestamp()
        ON MATCH SET
            b.seen_count = coalesce(b.seen_count, 0) + 1,
            b.last_seen = timestamp()

        MERGE (a)-[r:ALIAS_WITH]->(b)
        ON CREATE SET
            r.rule_score     = $rule_score,
            r.string_sim     = $string_sim,
            r.embed_sim      = $embed_sim,
            r.llm_score      = $llm_score,
            r.combined_score = $combined_score,
            r.sources        = $sources,
            r.seen_count     = 1
        ON MATCH SET
            // Prefer newest non-null scores, otherwise keep existing
            r.rule_score     = coalesce($rule_score, r.rule_score),
            r.string_sim     = coalesce($string_sim, r.string_sim),
            r.embed_sim      = coalesce($embed_sim, r.embed_sim),
            r.llm_score      = coalesce($llm_score, r.llm_score),
            r.combined_score = coalesce($combined_score, r.combined_score),
            // Append sources list (may contain duplicates need to be fixed)
            r.sources        = coalesce(r.sources, []) + $sources,
            r.seen_count     = coalesce(r.seen_count, 0) + 1
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
            # Neo4j expects a list, so making sure we give it one
            sources=pair.sources or [],
        )

    # ---------- READ / EXPORT ----------

    def export_graph_for_threshold(self, threshold: float) -> nx.Graph:
        """
        Exports an undirected graph of Value nodes for edges with
        combined_score >= threshold.

        - we ask Neo4j: "Give me all pairs (a,b) where the edge between
          them is strong enough (above threshold)."
        - Build a NetworkX Graph from those pairs.
        - That graph includes edges learned in previous runs, not just
          current run, because they persist in Neo4j.
        """
        G = nx.Graph()
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:Value)-[r:ALIAS_WITH]->(b:Value)
                WHERE r.combined_score >= $threshold
                RETURN a.value AS left, b.value AS right
                """,
                threshold=threshold,
            )
            for record in result:
                G.add_edge(record["left"], record["right"])
        return G

    def write_cluster_ids(self, cluster_map: Dict[str, int]) -> None:
        """
        Write cluster_id back onto nodes for explainability and reuse.

        - For each normalized value, set v.cluster_id = <component id>.
        - Next run, we can query these cluster_ids to instantly know
          which alias set a value belongs to.
        """
        with self.driver.session() as session:
            for value, cid in cluster_map.items():
                session.run(
                    """
                    MATCH (v:Value {value: $value})
                    SET v.cluster_id = $cid
                    """,
                    value=value,
                    cid=cid,
                )

    def get_cluster_ids_for_values(self, values: List[str]) -> Dict[str, int]:
        """
        Optional helper: given a list of normalized values, return any
        existing cluster_ids from Neo4j.

        - This lets the pipeline "reuse knowledge" by checking if some
          values are already in stable clusters from previous runs.
        """
        if not values:
            return {}

        mapping: Dict[str, int] = {}
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $vals AS v
                MATCH (n:Value {value: v})
                WHERE n.cluster_id IS NOT NULL
                RETURN n.value AS value, n.cluster_id AS cid
                """,
                vals=values,
            )
            for rec in result:
                mapping[rec["value"]] = rec["cid"]
        return mapping

from networkx.algorithms.community import label_propagation_communities

def build_graph_and_clusters(
    pairs: Dict[Tuple[str, str], CandidatePair],
    cfg: PipelineConfig,
) -> GraphStats:
    """
    Build the value alias graph in Neo4j and compute synonym clusters via
    connected components on high confidence edges.

    1. Make sure Neo4j has the right constraints / indexes.
    2. For every candidate pair, upsert nodes + edge with scores.
       This is where we PERSIST knowledge from this run.
    3. Asks Neo4j for the "strong" subgraph: edges with combined_score
       >= accept_threshold.
    4. Run NetworkX connected_components on that subgraph to find
       alias clusters.
    5. Write cluster ids back into Neo4j for reuse.
    6. Return simple stats + the cluster_map (value -> cluster_id) so
       the pipeline can materialize the final semantic join.
    """
    graph_stats = GraphStats()
    neo = Neo4jGraph(cfg)
    neo.init_schema()

    # 1) Write all pairs to Neo4j with their scores
    for pair in pairs.values():
        neo.upsert_pair(pair)

    # 2) Export accepted subgraph (combined_score >= accept_threshold)
    G = neo.export_graph_for_threshold(cfg.accept_threshold)

    # 3) Label propagation => communities / synonym sets
    cluster_map: Dict[str, int] = {}

    if G.number_of_nodes() > 0:
        communities = label_propagation_communities(G)
        for cluster_id, community in enumerate(communities, start=1):
            for value in community:
                cluster_map[value] = cluster_id

    # 4) Persist cluster ids on Neo4j nodes
    neo.write_cluster_ids(cluster_map)
    neo.close()

    # 5) Fill stats
    graph_stats.n_nodes = G.number_of_nodes()
    graph_stats.n_edges = G.number_of_edges()
    graph_stats.n_clusters = len({cid for cid in cluster_map.values()})
    graph_stats.cluster_map = cluster_map

    return graph_stats
