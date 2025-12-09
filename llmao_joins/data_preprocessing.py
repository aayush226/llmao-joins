import re
import unicodedata
from typing import List, Set, Tuple, Dict, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import Levenshtein
from config import PipelineConfig


class DataNormalizer:
    def __init__(self, custom_abbreviations: Optional[Dict[str, str]] = None):
        """
        Initialize normalizer with optional custom abbreviations
        
        Args:
            custom_abbreviations: Additional abbreviation mappings
        """
        self.abbreviations = self.ABBREVIATIONS.copy()

        # 1. Apply global master abbreviations (if provided)
        if PipelineConfig.abbrevation_master:
            self._load_abbreviation_file(PipelineConfig.abbrevation_master)

        # 2. Apply instance-level custom abbreviations (highest priority)
        if custom_abbreviations:
            self.abbreviations.update(custom_abbreviations)
    
    def _load_abbreviation_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.strip().split(",")
                self.abbreviations[key] = value
    
    def normalize(self, text: str) -> str:
        """
        Complete normalization pipeline
        
        Args:
            text: Raw input string
            
        Returns:
            Normalized string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Unicode normalization
        text = self._normalize_unicode(text)
        
        # Step 2: Lowercase
        text = text.lower()
        
        # Step 3: Remove leading "the"
        text = self._remove_leading_the(text)
        
        # Step 4: Remove punctuation and special chars
        text = self._remove_punctuation(text)
        
        # Step 5: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Step 6: Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Step 7: Strip
        text = text.strip()
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters (e.g., accents)"""
        # NFD = decomposed form, then filter out combining marks
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
    
    def _remove_leading_the(self, text: str) -> str:
        """Remove leading 'the' or 'a'"""
        text = text.strip()
        if text.startswith('the '):
            return text[4:]
        if text.startswith('a '):
            return text[2:]
        return text
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation but keep spaces"""
        # Keep alphanumeric and spaces
        return re.sub(r'[^\w\s]', ' ', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces into one"""
        return re.sub(r'\s+', ' ', text)
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand known abbreviations"""
        if not self.abbreviations:
            return text
        words = text.split()
        expanded = []
        
        for word in words:
            if word in self.abbreviations:
                expanded.append(self.abbreviations[word])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def normalize_dataframe_column(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Normalize an entire DataFrame column
        
        Args:
            df: Input DataFrame
            column: Column name to normalize
            
        Returns:
            Series of normalized values
        """
        return df[column].astype(str).apply(self.normalize)


class NGramGenerator:
    """Generate n-grams for blocking/candidate generation"""
    
    @staticmethod
    def generate_ngrams(text: str, n: int = 3) -> Set[str]:
        """
        Generate character n-grams from text
        
        Args:
            text: Input text
            n: N-gram size (default: 3 for trigrams)
            
        Returns:
            Set of n-grams
        """
        if len(text) < n:
            return {text}
        
        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i+n])
        
        return ngrams
    
    @staticmethod
    def generate_word_ngrams(text: str, n: int = 2) -> Set[str]:
        """
        Generate word-level n-grams
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            Set of word n-grams
        """
        words = text.split()
        if len(words) < n:
            return {text}
        
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngrams.add(' '.join(words[i:i+n]))
        
        return ngrams
    
    @staticmethod
    def jaccard_similarity(set1: Set, set2: Set) -> float:
        """
        Calculate Jaccard similarity between two sets
        
        Args:
            set1, set2: Sets to compare
            
        Returns:
            Jaccard similarity score [0, 1]
        """
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


class StringSimilarity:
    """Calculate various string similarity metrics"""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein edit distance
        
        Args:
            s1, s2: Strings to compare
            
        Returns:
            Edit distance (integer)
        """
        return Levenshtein.distance(s1, s2)
    
    @staticmethod
    def normalized_levenshtein(s1: str, s2: str) -> float:
        """
        Normalized Levenshtein distance [0, 1]
        0 = identical, 1 = completely different
        
        Args:
            s1, s2: Strings to compare
            
        Returns:
            Normalized distance
        """
        if not s1 and not s2:
            return 0.0
        
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 0.0
        
        distance = Levenshtein.distance(s1, s2)
        return distance / max_len
    
    @staticmethod
    def levenshtein_similarity(s1: str, s2: str) -> float:
        """
        Levenshtein similarity score [0, 1]
        1 = identical, 0 = completely different
        
        Args:
            s1, s2: Strings to compare
            
        Returns:
            Similarity score
        """
        return 1.0 - StringSimilarity.normalized_levenshtein(s1, s2)
    
    @staticmethod
    def jaro_winkler_similarity(s1: str, s2: str) -> float:
        """
        Calculate Jaro-Winkler similarity
        Better for short strings, gives more weight to common prefixes
        
        Args:
            s1, s2: Strings to compare
            
        Returns:
            Similarity score [0, 1]
        """
        return Levenshtein.jaro_winkler(s1, s2)
    
    @staticmethod
    def sequence_matcher_ratio(s1: str, s2: str) -> float:
        """
        Python's SequenceMatcher ratio (similar to difflib)
        
        Args:
            s1, s2: Strings to compare
            
        Returns:
            Similarity ratio [0, 1]
        """
        return SequenceMatcher(None, s1, s2).ratio()


class CandidatePairGenerator:
    """Generate candidate pairs using blocking and similarity thresholds"""
    
    def __init__(self, 
                 min_similarity: float = 0.6,
                 use_ngrams: bool = True,
                 ngram_size: int = 3,
                 use_edit_distance: bool = True,
                 max_edit_distance: int = 5):
        """
        Initialize candidate generator
        
        Args:
            min_similarity: Minimum similarity threshold for candidates
            use_ngrams: Whether to use n-gram blocking
            ngram_size: Size of n-grams for blocking
            use_edit_distance: Whether to filter by edit distance
            max_edit_distance: Maximum edit distance for candidates
        """
        self.min_similarity = min_similarity
        self.use_ngrams = use_ngrams
        self.ngram_size = ngram_size
        self.use_edit_distance = use_edit_distance
        self.max_edit_distance = max_edit_distance
        self.ngram_gen = NGramGenerator()
        self.string_sim = StringSimilarity()
    
    def generate_candidates_bruteforce(self, 
                                       values1: List[str], 
                                       values2: List[str]) -> List[Tuple[str, str, float]]:
        """
        Brute force candidate generation (for small datasets)
        Compares all pairs
        
        Args:
            values1: First list of values
            values2: Second list of values
            
        Returns:
            List of (value1, value2, similarity_score) tuples
        """
        candidates = []
        
        for v1 in values1:
            for v2 in values2:
                if v1 == v2:
                    candidates.append((v1, v2, 1.0))
                    continue
                
                # Calculate similarity
                sim = self._calculate_similarity(v1, v2)
                
                if sim >= self.min_similarity:
                    candidates.append((v1, v2, sim))
        
        return candidates
    
    def generate_candidates_with_blocking(self,
                                          values1: List[str],
                                          values2: List[str]) -> List[Tuple[str, str, float]]:
        """
        Generate candidates using n-gram blocking
        Much faster than brute force for large datasets
        
        Args:
            values1: First list of values
            values2: Second list of values
            
        Returns:
            List of (value1, value2, similarity_score) tuples
        """
        # Build inverted index: ngram -> list of values
        index = defaultdict(set)
        
        # Index values2
        for v2 in values2:
            ngrams = self.ngram_gen.generate_ngrams(v2, self.ngram_size)
            for ngram in ngrams:
                index[ngram].add(v2)
        
        candidates = []
        seen_pairs = set()
        
        # For each value in values1, find candidates from values2
        for v1 in values1:
            ngrams1 = self.ngram_gen.generate_ngrams(v1, self.ngram_size)
            
            # Find all values2 that share at least one n-gram
            candidate_values = set()
            for ngram in ngrams1:
                candidate_values.update(index[ngram])
            
            # Calculate similarity for each candidate
            for v2 in candidate_values:
                pair = tuple(sorted([v1, v2]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                sim = self._calculate_similarity(v1, v2)
                
                if sim >= self.min_similarity:
                    candidates.append((v1, v2, sim))
        
        return candidates
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate combined similarity score
        
        Args:
            s1, s2: Strings to compare
            
        Returns:
            Combined similarity score [0, 1]
        """
        # Exact match
        if s1 == s2:
            return 1.0
        
        # Edit distance filter
        if self.use_edit_distance:
            edit_dist = self.string_sim.levenshtein_distance(s1, s2)
            if edit_dist > self.max_edit_distance:
                return 0.0
        
        # Calculate multiple similarity metrics and average
        scores = []
        
        # Levenshtein similarity
        scores.append(self.string_sim.levenshtein_similarity(s1, s2))
        
        # Jaro-Winkler (good for short strings)
        scores.append(self.string_sim.jaro_winkler_similarity(s1, s2))
        
        # N-gram similarity
        if self.use_ngrams:
            ngrams1 = self.ngram_gen.generate_ngrams(s1, self.ngram_size)
            ngrams2 = self.ngram_gen.generate_ngrams(s2, self.ngram_size)
            scores.append(self.ngram_gen.jaccard_similarity(ngrams1, ngrams2))
        
        # Return average of all scores
        return np.mean(scores)
    
    def generate_self_join_candidates(self, values: List[str]) -> List[Tuple[str, str, float]]:
        """
        Generate candidates for self-join (finding duplicates within one list)
        
        Args:
            values: List of values to deduplicate
            
        Returns:
            List of (value1, value2, similarity_score) tuples
        """
        # Build inverted index
        index = defaultdict(set)
        
        for v in values:
            ngrams = self.ngram_gen.generate_ngrams(v, self.ngram_size)
            for ngram in ngrams:
                index[ngram].add(v)
        
        candidates = []
        seen_pairs = set()
        
        for v1 in values:
            ngrams1 = self.ngram_gen.generate_ngrams(v1, self.ngram_size)
            
            # Find candidates
            candidate_values = set()
            for ngram in ngrams1:
                candidate_values.update(index[ngram])
            
            for v2 in candidate_values:
                if v1 == v2:
                    continue
                
                # Avoid duplicate pairs
                pair = tuple(sorted([v1, v2]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                sim = self._calculate_similarity(v1, v2)
                
                if sim >= self.min_similarity:
                    candidates.append((v1, v2, sim))
        
        return candidates


class BKTree:
    """
    BK-Tree (Burkhard-Keller Tree) for efficient edit distance search
    Useful for finding all values within a given edit distance
    """
    
    class Node:
        def __init__(self, value: str):
            self.value = value
            self.children = {}
    
    def __init__(self):
        self.root = None
        self.string_sim = StringSimilarity()
    
    def add(self, value: str):
        """Add a value to the BK-Tree"""
        if self.root is None:
            self.root = self.Node(value)
            return
        
        current = self.root
        distance = self.string_sim.levenshtein_distance(value, current.value)
        
        while distance in current.children:
            current = current.children[distance]
            distance = self.string_sim.levenshtein_distance(value, current.value)
        
        current.children[distance] = self.Node(value)
    
    def search(self, value: str, max_distance: int) -> List[Tuple[str, int]]:
        """
        Search for all values within max_distance edits
        
        Args:
            value: Query value
            max_distance: Maximum edit distance
            
        Returns:
            List of (value, distance) tuples
        """
        if self.root is None:
            return []
        
        results = []
        
        def _search(node, query, max_dist):
            distance = self.string_sim.levenshtein_distance(query, node.value)
            
            if distance <= max_dist:
                results.append((node.value, distance))
            
            # Search children within the distance range
            for child_dist in range(distance - max_dist, distance + max_dist + 1):
                if child_dist in node.children:
                    _search(node.children[child_dist], query, max_dist)
        
        _search(self.root, value, max_distance)
        return results


class PairStatistics:
    """Calculate statistics on generated candidate pairs"""
    
    @staticmethod
    def calculate_statistics(candidates: List[Tuple[str, str, float]]) -> Dict:
        """
        Calculate statistics on candidate pairs
        
        Args:
            candidates: List of (value1, value2, score) tuples
            
        Returns:
            Dictionary of statistics
        """
        if not candidates:
            return {
                'num_pairs': 0,
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0,
                'median_similarity': 0.0
            }
        
        scores = [score for _, _, score in candidates]
        
        return {
            'num_pairs': len(candidates),
            'avg_similarity': np.mean(scores),
            'min_similarity': np.min(scores),
            'max_similarity': np.max(scores),
            'median_similarity': np.median(scores),
            'std_similarity': np.std(scores)
        }
    
    @staticmethod
    def filter_by_confidence(candidates: List[Tuple[str, str, float]],
                           high_threshold: float = 0.85,
                           low_threshold: float = 0.5) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Split candidates into confidence buckets
        
        Args:
            candidates: List of (value1, value2, score) tuples
            high_threshold: Threshold for high confidence matches
            low_threshold: Threshold for uncertain matches
            
        Returns:
            Dictionary with 'high_confidence', 'uncertain', 'low_confidence' lists
        """
        high_confidence = []
        uncertain = []
        low_confidence = []
        
        for v1, v2, score in candidates:
            if score >= high_threshold:
                high_confidence.append((v1, v2, score))
            elif score >= low_threshold:
                uncertain.append((v1, v2, score))
            else:
                low_confidence.append((v1, v2, score))
        
        return {
            'high_confidence': high_confidence,
            'uncertain': uncertain,
            'low_confidence': low_confidence
        }
    
def write_line(f, line: str = ""):
    f.write(line + "\n")

# Example usage and testing
if __name__ == "__main__":
    # Example data
    df1 = pd.read_csv("C:/Users/Bineet/Downloads/left.csv", dtype=str)
    col_name1 = df1.columns[1]
    df2 = pd.read_csv("C:/Users/Bineet/Downloads/right.csv", dtype=str)
    col_name2 = df2.columns[1]
    # if df.shape[1] != 1:
    #     raise ValueError(f"CSV at {path} should have exactly one column")
    # return df.iloc[:,0].astype(str).tolist()
    # values1 = [
    #     "USA",
    #     "The Netherlands",
    #     "U.K.",
    #     "South Korea",
    #     "UAE"
    # ]
    values1 = df1[col_name1].fillna('').astype(str).tolist()
    values2 = df2[col_name2].fillna('').astype(str).tolist()
    # values2 = [
    #     "United States of America",
    #     "Netherlands",
    #     "United Kingdom",
    #     "Republic of Korea",
    #     "United Arab Emirates"
    # ]
    
    # Step 1: Normalize
    print("=== Step 1: Normalization ===")
    normalizer = DataNormalizer()
    
    norm_values1 = [normalizer.normalize(v) for v in values1]
    norm_values2 = [normalizer.normalize(v) for v in values2]
    output_file = "output.txt"
    # optional: if you want a separate directory
    # os.makedirs("output", exist_ok=True)
    # output_file = os.path.join("output", "results.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        write_line(f, "=== Transformation Discovery ===")
        
        # Suppose trans_chain is determined earlier
        # if trans_chain:
        #     write_line(f, f"Transformation chain: {trans_chain}")
        # else:
        #     write_line(f, "Transformation chain: None")

        write_line(f, "")
        write_line(f, "Original -> Normalized:")
        for orig, norm in zip(values1, norm_values1):
            write_line(f, f"  {orig:30} -> {norm}")

        write_line(f, "")
        write_line(f, "=== Step 2: Candidate Generation ===")
        generator = CandidatePairGenerator(min_similarity=0.5)
        candidates = generator.generate_candidates_with_blocking(norm_values1, norm_values2)

        write_line(f, f"\nFound {len(candidates)} candidate pairs:")
        for v1, v2, score in sorted(candidates, key=lambda x: x[2], reverse=True):
            write_line(f, f"  {v1:30} <-> {v2:30} (score: {score:.3f})")

        write_line(f, "")
        write_line(f, "=== Step 3: Statistics ===")
        stats = PairStatistics.calculate_statistics(candidates)

        write_line(f, "\nCandidate Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                write_line(f, f"  {key:20}: {value:.3f}")
            else:
                write_line(f, f"  {key:20}: {value}")

        write_line(f, "")
        write_line(f, "=== Step 4: Confidence Filtering ===")
        filtered = PairStatistics.filter_by_confidence(candidates)

        write_line(f, f"\nHigh confidence ({len(filtered['high_confidence'])} pairs):")
        for v1, v2, score in filtered['high_confidence']:
            write_line(f, f"  {v1:30} <-> {v2:30} (score: {score:.3f})")

        write_line(f, "")
        write_line(f, f"\nUncertain ({len(filtered['uncertain'])} pairs) - these would go to LLM:")
        for v1, v2, score in filtered['uncertain']:
            write_line(f, f"  {v1:30} <-> {v2:30} (score: {score:.3f})")

    print(f"Results written to {output_file}")
    # print("\nOriginal -> Normalized:")
    # for orig, norm in zip(values1, norm_values1):
    #     print(f"  {orig:30} -> {norm}")
    
    # # Step 2: Generate candidates
    # print("\n=== Step 2: Candidate Generation ===")
    # generator = CandidatePairGenerator(min_similarity=0.5)
    
    # candidates = generator.generate_candidates_with_blocking(norm_values1, norm_values2)
    
    # print(f"\nFound {len(candidates)} candidate pairs:")
    # for v1, v2, score in sorted(candidates, key=lambda x: x[2], reverse=True):
    #     print(f"  {v1:30} <-> {v2:30} (score: {score:.3f})")
    
    # # Step 3: Statistics
    # print("\n=== Step 3: Statistics ===")
    # stats = PairStatistics.calculate_statistics(candidates)
    # print(f"\nCandidate Statistics:")
    # for key, value in stats.items():
    #     print(f"  {key:20}: {value:.3f}" if isinstance(value, float) else f"  {key:20}: {value}")
    
    # # Step 4: Filter by confidence
    # print("\n=== Step 4: Confidence Filtering ===")
    # filtered = PairStatistics.filter_by_confidence(candidates)
    
    # print(f"\nHigh confidence ({len(filtered['high_confidence'])} pairs):")
    # for v1, v2, score in filtered['high_confidence']:
    #     print(f"  {v1:30} <-> {v2:30} (score: {score:.3f})")
    
    # print(f"\nUncertain ({len(filtered['uncertain'])} pairs) - these would go to LLM:")
    # for v1, v2, score in filtered['uncertain']:
    #     print(f"  {v1:30} <-> {v2:30} (score: {score:.3f})")