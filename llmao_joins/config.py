# file: llmao_joins/config.py
import os
from dataclasses import dataclass
from typing import Tuple, Optional
from dotenv import load_dotenv


load_dotenv()

@dataclass
class PipelineConfig:
    # IO
    left_csv: str
    right_csv: str
    left_col: str
    right_col: str
    abbrevation_master: Optional[str] = None 
    output_dir: str = "outputs"

    # Candidate generation limits
    max_unique_values: int = 50000  # safety cap

    # String-based candidate generation
    ngram_range: Tuple[int, int] = (3, 3)
    string_top_k: int = 20
    string_min_sim: float = 0.6

    # Embedding-based candidate generation
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_top_k: int = 20
    embed_min_sim: float = 0.75

    # Scoring weights (how we combine multiple similarity measures)
    w_rule: float = 0.6
    w_string: float = 0.2
    w_embed: float = 0.2
    w_llm: float = 0.7
    w_low: float = 0.5
    w_high: float = 0.8
    w_ngram_jaccard: float = 0.0
    w_levenshtein: float = 0.0
    w_jaro_winkler: float = 0.0
    w_minhash: float = 0.0
    # Acceptance / LLM band
    accept_threshold: float = 0.55
    llm_band_low: float = 0.4
    llm_band_high: float = 0.8
    max_llm_queries: int = 500

    # Neo4j connection
    neo4j_uri: str = os.getenv("NEO4J_URI")
    neo4j_user: str = os.getenv("NEO4J_USER")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD")

    # LLM cost (for logging only)
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")
    llm_model: str = "gpt-4o-mini"
    llm_price_per_1k_tokens: float = 0.15
