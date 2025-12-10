# file: llmao_joins/scoring_and_llm.py
import openai
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
    Combining all similarity measures into a single score using configurable weights.
    """
    f = pair.features or {}

    # Collect (score, weight) only for signals we actually have
    scores = []
    weights = []

    if pair.rule_score is not None:
        scores.append(pair.rule_score)
        weights.append(cfg.w_rule)

    if pair.string_sim is not None:
        scores.append(pair.string_sim)
        weights.append(cfg.w_string)

    if pair.embed_sim is not None:
        scores.append(pair.embed_sim)
        weights.append(cfg.w_embed)

    if pair.llm_score is not None:
        scores.append(pair.llm_score)
        weights.append(cfg.w_llm)

    # feature-based scores
    if "levenshtein_sim" in f:
        scores.append(f["levenshtein_sim"])
        weights.append(cfg.w_levenshtein)

    if "jaro_winkler" in f:
        scores.append(f["jaro_winkler"])
        weights.append(cfg.w_jaro_winkler)

    if "ngram_jaccard" in f:
        scores.append(f["ngram_jaccard"])
        weights.append(cfg.w_ngram_jaccard)

    if "minhash" in f:
        scores.append(f["minhash"])
        weights.append(cfg.w_minhash)

    if not scores:
        pair.combined_score = 0.0
        return 0.0

    raw = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    pair.combined_score = max(0.0, min(1.0, raw))
    return pair.combined_score

def score_all(pairs: Dict[Tuple[str, str], CandidatePair], cfg: PipelineConfig) -> None:
    for p in pairs.values():
        score_candidate(p, cfg)

def _approx_token_count(text: str) -> int:
    # Very rough heuristic: 4 characters per token
    return max(1, int(len(text) / 4))

from .llm_client import ask_if_synonyms, set_openai_key
import random

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
    stats = LLMStats()
    if not api_key:
        print("[LLM GATE] Skipping – No API key provided.")
        return stats
    
    set_openai_key(api_key)
    model = cfg.llm_model

    count = 0
    candidates = [
        pair for pair in pairs.values()
        if cfg.llm_band_low <= (pair.combined_score or 0.0) <= cfg.llm_band_high
        # skip trivial perfect self-matches
        and not (pair.rule_score == 1.0 and pair.left_norm == pair.right_norm)
    ]
    random.shuffle(candidates)

    for pair in candidates:
        if count >= cfg.max_llm_queries:
            break

        answer = ask_if_synonyms(pair.left_norm, pair.right_norm, model=model)

        if answer == "YES":
            pair.llm_score = 1.0
        elif answer == "NO":
            pair.llm_score = 0.0   # penalize
        else:
            continue  # UNKNOWN → no change

        # (optional but nice) mark that LLM was used
        if "llm" not in pair.sources:
            pair.sources.append("llm")

        count += 1
        prompt_tokens = _approx_token_count(pair.left_norm + pair.right_norm)
        stats.n_calls += 1
        stats.total_prompt_tokens += prompt_tokens
        stats.total_completion_tokens += 1

    print(f"[LLM GATE] Queried {count} pairs using {model}")
    return stats
