# file: llmao_joins/candidate_generation.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .io_and_normalization import ValueRecord
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import numpy as np

@dataclass
class CandidatePair:
    left_raw: str
    right_raw: str
    left_norm: str
    right_norm: str
    rule_score: Optional[float] = None
    string_sim: Optional[float] = None
    embed_sim: Optional[float] = None
    llm_score: Optional[float] = None
    combined_score: Optional[float] = None
    features: Dict[str, float] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)

    def key(self) -> Tuple[str, str]:
        return (self.left_raw, self.right_raw)

def _get_or_create(
    pairs: Dict[Tuple[str, str], CandidatePair],
    left: ValueRecord,
    right: ValueRecord,
) -> CandidatePair:
    key = (left.raw, right.raw)
    if key not in pairs:
        pairs[key] = CandidatePair(
            left_raw=left.raw,
            right_raw=right.raw,
            left_norm=left.norm,
            right_norm=right.norm,
        )
    return pairs[key]

def generate_rule_pairs(
    left_values: List[ValueRecord],
    right_values: List[ValueRecord],
    pairs: Optional[Dict[Tuple[str, str], CandidatePair]] = None,
    threshold: int = 75,  # can change
) -> Dict[Tuple[str, str], CandidatePair]:
    """
    Step 2a: High-precision rule-based candidates.
    - Inputs: normalized values from left/right join columns.
    - Pruning: only when canonical forms match.
    """
    if pairs is None:
        pairs = {}

    # Index by normalized value for fast cross-joins
    left_by_norm: Dict[str, List[ValueRecord]] = {}
    right_by_norm: Dict[str, List[ValueRecord]] = {}

    for v in left_values:
        left_by_norm.setdefault(v.norm, []).append(v)
    for v in right_values:
        right_by_norm.setdefault(v.norm, []).append(v)

    shared_norms = set(left_by_norm.keys()) & set(right_by_norm.keys())

    for norm in shared_norms:
        for l in left_by_norm[norm]:
            for r in right_by_norm[norm]:
                pair = _get_or_create(pairs, l, r)
                pair.rule_score = 1.0
                if "rule" not in pair.sources:
                    pair.sources.append("rule")

    return pairs

def generate_string_candidates(
    left_values: List[ValueRecord],
    right_values: List[ValueRecord],
    ngram_range=(3, 3),
    top_k: int = 20,
    min_sim: float = 0.6,
    pairs: Optional[Dict[Tuple[str, str], CandidatePair]] = None,
) -> Dict[Tuple[str, str], CandidatePair]:
    """
    Step 2b: String-based candidates via char n-gram TF-IDF + cosine similarity.
    - Pruning:
      * only top_k nearest neighbors from right for each left value
      * discard below min_sim
    """
    if pairs is None:
        pairs = {}

    left_norms = [v.norm for v in left_values]
    right_norms = [v.norm for v in right_values]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(left_norms + right_norms)

    left_vecs = tfidf_matrix[: len(left_norms)]
    right_vecs = tfidf_matrix[len(left_norms) :]

    n_neighbors = min(top_k, len(right_values))
    if n_neighbors == 0:
        return pairs  # nothing to match against

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(right_vecs)
    distances, indices = nn.kneighbors(left_vecs, return_distance=True)

    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        left = left_values[i]
        for dist, j in zip(dist_row, idx_row):
            sim = 1.0 - dist
            if sim < min_sim:
                continue

            right = right_values[j]
            pair = _get_or_create(pairs, left, right)

            # Keep the best string similarity we've seen
            # (so we don't downgrade graph_prior = 1.0)
            if pair.string_sim is None or sim > pair.string_sim:
                pair.string_sim = sim
                pair.features["string_sim"] = sim

            # Avoid duplicate "string" entries
            if "string" not in pair.sources:
                pair.sources.append("string")

    return pairs

def generate_embedding_candidates(
    left_values: List[ValueRecord],
    right_values: List[ValueRecord],
    model_name: str,
    top_k: int = 20,
    min_sim: float = 0.6,
    pairs: Optional[Dict[Tuple[str, str], CandidatePair]] = None,
) -> Dict[Tuple[str, str], CandidatePair]:
    """
    Step 2c: Semantic candidates via SentenceTransformer embeddings.

    - Uses SentenceTransformer to encode left/right normalized values.
    - Uses util.semantic_search to find the top_k most similar right values
      for each left value (cosine similarity).
    - Only keeps pairs with similarity >= min_sim.
    """
    if pairs is None:
        pairs = {}

    # Nothing to do if either side is empty
    if not left_values or not right_values:
        return pairs

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Prepare normalized texts
    left_norms = [v.norm for v in left_values]
    right_norms = [v.norm for v in right_values]

    # Encode as tensors for semantic_search
    left_embeds = model.encode(
        left_norms,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine similarity becomes dot product sincr we normalise
    )
    right_embeds = model.encode(
        right_norms,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # Cap top_k so we don't ask for more neighbors than we have
    top_k = min(top_k, len(right_values))
    if top_k <= 0:
        return pairs

    # Perform semantic search: for each left embedding, get top_k right neighbors
    search_results = util.semantic_search(
        query_embeddings=left_embeds,
        corpus_embeddings=right_embeds,
        top_k=top_k,
    )

    for i, hits in enumerate(search_results):
        left = left_values[i]
        for hit in hits:
            j = hit["corpus_id"]
            sim = float(hit["score"])

            # Filter by minimum similarity
            if sim < min_sim:
                continue

            right = right_values[j]
            pair = _get_or_create(pairs, left, right)

            # Keep the best embedding similarity we've seen for this pair
            if pair.embed_sim is None or sim > pair.embed_sim:
                pair.embed_sim = sim
                pair.features["embed_sim"] = sim

            if "embed" not in pair.sources:
                pair.sources.append("embed")
    return pairs
