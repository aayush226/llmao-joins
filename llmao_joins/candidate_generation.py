# file: llmao_joins/candidate_generation.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .io_and_normalization import ValueRecord, canonical_form

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
) -> Dict[Tuple[str, str], CandidatePair]:
    """
    Step 2a: High-precision rule-based candidates.
    - Inputs: normalized values from left/right join columns.
    - Pruning: only when canonical forms match.
    """
    # TODO: implement rule based candidate generation
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
    return pairs
