# file: llmao_joins/io_and_normalization.py
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from . import data_preprocessing

_WHITESPACE_RE = re.compile(r"\s+")

# Very small demo abbreviation map. Extend for better recall.
ABBREVIATION_MAP: Dict[str, str] = {
    "usa": "united states of america",
    "u.s.": "united states of america",
    "u.s": "united states of america",
    "us": "united states of america",
    "uk": "united kingdom",
    "uae": "united arab emirates",
    "south korea": "republic of korea",
    "s. korea": "republic of korea",
}

@dataclass(frozen=True)
class ValueRecord:
    raw: str
    norm: str
    side: str  # 'left' or 'right'

def normalize_text(value: str) -> str:
    # text = data_preprocessing.DataNormalizer.normalize(value)
    normalizer = data_preprocessing.DataNormalizer()
    text = normalizer.normalize(value)
    return text

def canonical_form(norm: str) -> str:
    # Expand abbreviations like "usa" -> "united states of america"
    return ABBREVIATION_MAP.get(norm, norm)

def load_column_values(csv_path: str, column: str, side: str) -> Tuple[List[ValueRecord], pd.DataFrame]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise KeyError(f"Column {column!r} not found in {csv_path}")
    series = df[column].fillna("").astype(str)
    records: List[ValueRecord] = []
    for raw in series.unique():
        norm = normalize_text(raw)
        canon = canonical_form(norm)  # we missed to apply abbreviation or canonical mapping here. the rules are trvial but we cant generalise to universal case. so just plugging in.
        records.append(ValueRecord(raw=raw, norm=canon, side=side))
    return records, df


def build_lookup(records: List[ValueRecord]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Returns:
        norm_to_raws: norm -> list of raw strings that map to it
        raw_to_norm: raw -> norm
    """
    norm_to_raws: Dict[str, List[str]] = {}
    raw_to_norm: Dict[str, str] = {}
    for r in records:
        norm_to_raws.setdefault(r.norm, []).append(r.raw)
        raw_to_norm[r.raw] = r.norm
    return norm_to_raws, raw_to_norm