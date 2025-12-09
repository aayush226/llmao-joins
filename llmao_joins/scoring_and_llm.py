# file: llmao_joins/scoring_and_llm.py
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .candidate_generation import CandidatePair
from .config import PipelineConfig

@dataclass
class LLMStats:
    n_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def estimated_cost(self, price_per_1k: float) -> float:
        return (self.total_tokens / 1000.0) * price_per_1k

def score_candidate(pair: CandidatePair, cfg: PipelineConfig) -> float:
    """
    Combine all similarity measures into a single score using configurable weights.
    """
    # TODO: implement scoring combination
    pair.combined_score = 0.0
    return 0.0

def score_all(pairs: Dict[Tuple[str, str], CandidatePair], cfg: PipelineConfig) -> None:
    for p in pairs.values():
        score_candidate(p, cfg)

def _approx_token_count(text: str) -> int:
    # Very rough heuristic: 4 characters per token
    return max(1, int(len(text) / 4))

def run_llm_gate(
    pairs: Dict[Tuple[str, str], CandidatePair],
    cfg: PipelineConfig,
    api_key: str,
) -> LLMStats:
    """
    For pairs whose combined score falls into [llm_band_low, llm_band_high],
    query an LLM once with a strict YES/NO question.

    This gives:
      - better precision on borderline cases
      - explicit accounting of token usage and cost
    """
    # TODO: to implement LLM gate
    return LLMStats()
