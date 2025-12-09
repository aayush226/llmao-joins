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
    # TODO: implement rule based candidate generation
    if pairs is None:
        pairs = {}

    for l in left_values:
        for r in right_values:
            score = fuzz.token_sort_ratio(l.norm, r.norm)
            if score >= threshold:
                pair = _get_or_create(pairs, l, r)
                pair.rule_score = 1.0
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
    # TODO: implement TF-IDF + NN candidate generation
    if pairs is None:
        pairs = {}

    left_norms = [v.norm for v in left_values]
    right_norms = [v.norm for v in right_values]

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(left_norms + right_norms)

    left_vecs = tfidf_matrix[:len(left_norms)]
    right_vecs = tfidf_matrix[len(left_norms):]

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
            if sim >= min_sim:
                right = right_values[j]
                pair = _get_or_create(pairs, left, right)
                pair.string_sim = sim
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
    - Pruning:
      * only top_k semantic neighbors per left value
      * discard below min_sim
    """
    # TODO: implement embedding-based candidate generation
    if pairs is None:
        pairs = {}

    model = SentenceTransformer(model_name)
    left_norms = [v.norm for v in left_values]
    right_norms = [v.norm for v in right_values]

    left_embeds = model.encode(left_norms, convert_to_tensor=True, show_progress_bar=False)
    right_embeds = model.encode(right_norms, convert_to_tensor=True, show_progress_bar=False)

    sim_matrix = util.cos_sim(left_embeds, right_embeds).cpu().numpy()

    top_k = min(top_k, len(right_values)) 
    for i, row in enumerate(sim_matrix):
        safe_k = min(top_k, len(row))
        if safe_k == 0:
            continue  # Skip if no candidates
        safe_k = min(safe_k, len(row) - 1)  # Avoid out-of-bounds
        top_indices = np.argpartition(-row, safe_k)[:safe_k]
        for j in top_indices:
            sim = row[j]
            if sim >= min_sim:
                left = left_values[i]
                right = right_values[j]
                pair = _get_or_create(pairs, left, right)
                pair.embed_sim = sim
                pair.sources.append("embed")
    return pairs
