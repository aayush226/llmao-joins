"""
Advanced Value Matching Techniques
Team: LLMao Joins

Inspired by:
- Auto-FuzzyJoin (SIGMOD 2021): Multiple join configurations
- Auto-Join (VLDB 2017): Transformation-based matching
- Distribution-based Matching (SIGMOD 2011): EMD for numerical data
- Valentine Framework (ICDE 2021): Comprehensive benchmarking
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Callable
from dataclasses import dataclass
from collections import Counter, defaultdict
from scipy.stats import wasserstein_distance
import pandas as pd


@dataclass
class JoinConfiguration:
    """
    Join configuration as in Auto-FuzzyJoin
    Defines preprocessing, tokenization, weighting, and distance function
    """
    name: str
    preprocessing: List[str]  # ['lowercase', 'remove_punctuation', etc.]
    tokenization: str  # 'words', 'qgrams', 'chars'
    token_weights: str  # 'equal', 'idf', 'frequency'
    distance_function: str  # 'levenshtein', 'jaccard', 'cosine'
    threshold: float
    
    def __repr__(self):
        return f"Config({self.name}, thresh={self.threshold})"


class ValueTransformation:
    """
    Transformation-based matching inspired by Auto-Join (VLDB 2017)
    Discovers transformation functions between value sets
    """
    
    # Common transformation operations
    TRANSFORMATIONS = {
        'lowercase': lambda x: x.lower(),
        'uppercase': lambda x: x.upper(),
        'strip': lambda x: x.strip(),
        'remove_spaces': lambda x: x.replace(' ', ''),
        'remove_punctuation': lambda x: re.sub(r'[^\w\s]', '', x),
        'first_char': lambda x: x[0] if x else '',
        'last_char': lambda x: x[-1] if x else '',
        'first_word': lambda x: x.split()[0] if x.split() else '',
        'last_word': lambda x: x.split()[-1] if x.split() else '',
        'initials': lambda x: ''.join([w[0] for w in x.split() if w]),
        'remove_the': lambda x: re.sub(r'^the\s+', '', x, flags=re.IGNORECASE),
        'alphanumeric_only': lambda x: re.sub(r'[^a-zA-Z0-9]', '', x),
    }
    
    @staticmethod
    def discover_transformation(source_values: List[str],
                               target_values: List[str],
                               max_depth: int = 2) -> Optional[List[str]]:
        """
        Discover transformation sequence from source to target
        
        Args:
            source_values: Source value set
            target_values: Target value set
            max_depth: Maximum transformation chain length
            
        Returns:
            List of transformation names or None
        """
        target_set = set(target_values)
        
        # Try single transformations
        for trans_name, trans_func in ValueTransformation.TRANSFORMATIONS.items():
            try:
                transformed = [trans_func(v) for v in source_values]
                matches = sum(1 for t in transformed if t in target_set)
                match_ratio = matches / len(source_values)
                
                if match_ratio > 0.8:  # 80% match threshold
                    return [trans_name]
            except:
                continue
        
        # Try two-step transformations
        if max_depth >= 2:
            for trans1_name, trans1_func in ValueTransformation.TRANSFORMATIONS.items():
                for trans2_name, trans2_func in ValueTransformation.TRANSFORMATIONS.items():
                    try:
                        transformed = [trans2_func(trans1_func(v)) for v in source_values]
                        matches = sum(1 for t in transformed if t in target_set)
                        match_ratio = matches / len(source_values)
                        
                        if match_ratio > 0.8:
                            return [trans1_name, trans2_name]
                    except:
                        continue
        
        return None
    
    @staticmethod
    def apply_transformation_chain(value: str, 
                                   transformations: List[str]) -> str:
        """Apply a chain of transformations to a value"""
        result = value
        for trans_name in transformations:
            if trans_name in ValueTransformation.TRANSFORMATIONS:
                result = ValueTransformation.TRANSFORMATIONS[trans_name](result)
        return result
    
    @staticmethod
    def numeric_transformation_pattern(source_values: List[str],
                                      target_values: List[str]) -> Optional[Dict]:
        """
        Discover numeric transformation patterns
        E.g., "1200000" -> "$1.2 M"
        
        Returns:
            Dictionary with transformation parameters or None
        """
        try:
            # Try to parse both as numbers
            source_nums = []
            target_nums = []
            
            for s, t in zip(source_values, target_values):
                # Extract numbers from strings
                s_num = float(re.sub(r'[^\d.]', '', s))
                t_num = float(re.sub(r'[^\d.]', '', t))
                
                source_nums.append(s_num)
                target_nums.append(t_num)
            
            # Check for scaling factor
            ratios = [s / t for s, t in zip(source_nums, target_nums) if t != 0]
            
            if len(set(ratios)) == 1 or np.std(ratios) < 0.01:
                # Consistent scaling factor found
                scale_factor = ratios[0]
                
                # Detect prefix/suffix pattern
                prefix = re.match(r'^([^\d]*)', target_values[0])
                suffix = re.search(r'([^\d]*)$', target_values[0])
                
                return {
                    'type': 'numeric_scaling',
                    'scale_factor': scale_factor,
                    'prefix': prefix.group(1) if prefix else '',
                    'suffix': suffix.group(1) if suffix else ''
                }
        except:
            pass
        
        return None


class DistributionBasedMatching:
    """
    Distribution-based matching inspired by Zhang et al. (SIGMOD 2011)
    Uses Earth Mover's Distance (EMD) for numerical and categorical data
    """
    
    @staticmethod
    def compute_emd_numerical(values1: List[float],
                             values2: List[float]) -> float:
        """
        Compute Earth Mover's Distance for numerical distributions
        
        Args:
            values1, values2: Numerical value lists
            
        Returns:
            EMD distance
        """
        return wasserstein_distance(values1, values2)
    
    @staticmethod
    def compute_emd_categorical(values1: List[str],
                               values2: List[str]) -> float:
        """
        Compute EMD for categorical data using rank transformation
        As described in Zhang et al. (SIGMOD 2011)
        
        Args:
            values1, values2: Categorical value lists
            
        Returns:
            EMD distance
        """
        # Get all unique values and sort them
        all_values = sorted(set(values1 + values2))
        
        # Create rank mapping
        rank_map = {val: i for i, val in enumerate(all_values)}
        
        # Transform to ranks
        ranks1 = [rank_map[v] for v in values1]
        ranks2 = [rank_map[v] for v in values2]
        
        # Compute EMD on ranks
        return wasserstein_distance(ranks1, ranks2)
    
    @staticmethod
    def are_columns_similar_by_distribution(col1: pd.Series,
                                           col2: pd.Series,
                                           threshold: float = 0.3) -> bool:
        """
        Check if two columns are similar based on distribution
        
        Args:
            col1, col2: Pandas Series
            threshold: Maximum EMD for similarity
            
        Returns:
            True if similar
        """
        # Determine if numerical or categorical
        is_numeric1 = pd.api.types.is_numeric_dtype(col1)
        is_numeric2 = pd.api.types.is_numeric_dtype(col2)
        
        if is_numeric1 and is_numeric2:
            emd = DistributionBasedMatching.compute_emd_numerical(
                col1.dropna().tolist(),
                col2.dropna().tolist()
            )
        else:
            # Treat as categorical
            emd = DistributionBasedMatching.compute_emd_categorical(
                col1.astype(str).tolist(),
                col2.astype(str).tolist()
            )
        
        return emd < threshold


class MultiConfigurationMatcher:
    """
    Multi-configuration fuzzy join inspired by Auto-FuzzyJoin (SIGMOD 2021)
    Tries multiple join configurations to find optimal matches
    """
    
    def __init__(self, min_precision: float = 0.8):
        """
        Initialize multi-configuration matcher
        
        Args:
            min_precision: Minimum precision constraint
        """
        self.min_precision = min_precision
        self.configurations = self._generate_configurations()
    
    def _generate_configurations(self) -> List[JoinConfiguration]:
        """Generate diverse join configurations"""
        configs = []
        
        # Configuration 1: Strict string matching
        configs.append(JoinConfiguration(
            name="strict",
            preprocessing=['lowercase', 'strip'],
            tokenization='words',
            token_weights='equal',
            distance_function='levenshtein',
            threshold=0.9
        ))
        
        # Configuration 2: Fuzzy with punctuation removal
        configs.append(JoinConfiguration(
            name="fuzzy_clean",
            preprocessing=['lowercase', 'remove_punctuation', 'strip'],
            tokenization='words',
            token_weights='equal',
            distance_function='levenshtein',
            threshold=0.75
        ))
        
        # Configuration 3: Character n-grams
        configs.append(JoinConfiguration(
            name="ngram",
            preprocessing=['lowercase'],
            tokenization='qgrams',
            token_weights='equal',
            distance_function='jaccard',
            threshold=0.6
        ))
        
        # Configuration 4: Initials matching
        # configs.append(JoinConfiguration(
        #     name="initials",
        #     preprocessing=['lowercase', 'initials'],
        #     tokenization='chars',
        #     token_weights='equal',
        #     distance_function='levenshtein',
        #     threshold=0.8
        # ))
        
        # Configuration 5: Very fuzzy (low threshold)
        configs.append(JoinConfiguration(
            name="very_fuzzy",
            preprocessing=['lowercase', 'alphanumeric'],
            tokenization='words',
            token_weights='idf',
            distance_function='jaccard',
            threshold=0.5
        ))
        
        return configs
    
    def find_matches_with_config(self,
                                values1: List[str],
                                values2: List[str],
                                config: JoinConfiguration) -> List[Tuple[str, str, float]]:
        """
        Find matches using a specific configuration
        
        Args:
            values1, values2: Value lists
            config: Join configuration
            
        Returns:
            List of (value1, value2, score) matches
        """
        # Apply preprocessing to both value sets
        processed1 = self._preprocess(values1, config.preprocessing)
        processed2 = self._preprocess(values2, config.preprocessing)
        
        matches = []
        
        # Simple pairwise comparison with configured distance
        from data_preprocessing import StringSimilarity
        sim = StringSimilarity()
        
        for i, (v1_orig, v1_proc) in enumerate(zip(values1, processed1)):
            for j, (v2_orig, v2_proc) in enumerate(zip(values2, processed2)):
                # Calculate similarity based on config
                if config.distance_function == 'levenshtein':
                    score = sim.levenshtein_similarity(v1_proc, v2_proc)
                elif config.distance_function == 'jaccard':
                    # Use n-gram Jaccard
                    from data_preprocessing import NGramGenerator
                    ng = NGramGenerator()
                    ngrams1 = ng.generate_ngrams(v1_proc, 3)
                    ngrams2 = ng.generate_ngrams(v2_proc, 3)
                    score = ng.jaccard_similarity(ngrams1, ngrams2)
                else:
                    score = sim.jaro_winkler_similarity(v1_proc, v2_proc)
                
                if score >= config.threshold:
                    matches.append((v1_orig, v2_orig, score))
        
        return matches
    
    def _preprocess(self, values: List[str], preprocessing: List[str]) -> List[str]:
        """Apply preprocessing steps"""
        result = values.copy()
        
        for step in preprocessing:
            if step in ValueTransformation.TRANSFORMATIONS:
                result = [ValueTransformation.TRANSFORMATIONS[step](v) for v in result]
        
        return result
    
    def find_optimal_configurations(self,
                                   values1: List[str],
                                   values2: List[str],
                                   ground_truth: Optional[Set[Tuple[str, str]]] = None) -> List[Tuple[JoinConfiguration, List]]:
        """
        Find optimal join configurations
        
        Args:
            values1, values2: Value lists
            ground_truth: Optional ground truth for evaluation
            
        Returns:
            List of (config, matches) tuples sorted by quality
        """
        results = []
        
        for config in self.configurations:
            matches = self.find_matches_with_config(values1, values2, config)
            
            # Evaluate if ground truth provided
            if ground_truth:
                match_set = {(m[0], m[1]) for m in matches}
                tp = len(match_set & ground_truth)
                fp = len(match_set - ground_truth)
                fn = len(ground_truth - match_set)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Only keep configs meeting precision constraint
                if precision >= self.min_precision:
                    results.append((config, matches, f1))
            else:
                # No ground truth, keep all
                results.append((config, matches, len(matches)))
        
        # Sort by F1 or number of matches
        results.sort(key=lambda x: x[2], reverse=True)
        
        return [(config, matches) for config, matches, _ in results]


class TokenWeighting:
    """
    Token weighting schemes for value matching
    Inspired by Auto-FuzzyJoin's token weighting strategies
    """
    
    @staticmethod
    def compute_idf_weights(value_sets: List[List[str]]) -> Dict[str, float]:
        """
        Compute IDF (Inverse Document Frequency) weights
        
        Args:
            value_sets: List of value lists (each list is a "document")
            
        Returns:
            Dictionary mapping tokens to IDF weights
        """
        import math
        
        # Count document frequency for each token
        doc_freq = defaultdict(int)
        total_docs = len(value_sets)
        
        for value_set in value_sets:
            unique_tokens = set()
            for value in value_set:
                tokens = value.lower().split()
                unique_tokens.update(tokens)
            
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Compute IDF
        idf_weights = {}
        for token, freq in doc_freq.items():
            idf_weights[token] = math.log(total_docs / freq)
        
        return idf_weights
    
    @staticmethod
    def compute_frequency_weights(values: List[str]) -> Dict[str, float]:
        """
        Compute frequency-based weights (more frequent = higher weight)
        
        Args:
            values: List of values
            
        Returns:
            Dictionary mapping tokens to frequency weights
        """
        token_counts = Counter()
        
        for value in values:
            tokens = value.lower().split()
            token_counts.update(tokens)
        
        # Normalize to [0, 1]
        max_count = max(token_counts.values()) if token_counts else 1
        
        return {token: count / max_count for token, count in token_counts.items()}
    
    @staticmethod
    def weighted_jaccard_similarity(tokens1: List[str],
                                   tokens2: List[str],
                                   weights: Dict[str, float]) -> float:
        """
        Compute weighted Jaccard similarity
        
        Args:
            tokens1, tokens2: Token lists
            weights: Token weight dictionary
            
        Returns:
            Weighted Jaccard similarity
        """
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection_weight = sum(weights.get(t, 1.0) for t in set1 & set2)
        union_weight = sum(weights.get(t, 1.0) for t in set1 | set2)
        
        return intersection_weight / union_weight if union_weight > 0 else 0.0


class HybridValueMatcher:
    """
    Hybrid value matcher combining multiple techniques
    Inspired by Valentine framework's comprehensive approach
    """
    
    def __init__(self):
        self.multi_config = MultiConfigurationMatcher()
        self.dist_matcher = DistributionBasedMatching()
        self.transformer = ValueTransformation()
    
    def find_matches(self,
                    col1: pd.Series,
                    col2: pd.Series,
                    methods: List[str] = ['transformation', 'multi_config', 'distribution']) -> Dict:
        """
        Find matches using multiple methods
        
        Args:
            col1, col2: Pandas Series to match
            methods: List of methods to use
            
        Returns:
            Dictionary with results from each method
        """
        results = {}
        
        values1 = col1.astype(str).tolist()
        values2 = col2.astype(str).tolist()
        
        # Method 1: Transformation discovery
        if 'transformation' in methods:
            trans_chain = self.transformer.discover_transformation(values1, values2)
            if trans_chain:
                results['transformation'] = {
                    'found': True,
                    'chain': trans_chain,
                    'matches': self._apply_transformation_matches(values1, values2, trans_chain)
                }
            else:
                results['transformation'] = {'found': False}
        
        # Method 2: Multi-configuration
        # if 'multi_config' in methods:
        #     configs_results = self.multi_config.find_optimal_configurations(values1, values2)
            # if configs_results:
            #     best_config, best_matches = configs_results[0]
            #     results['multi_config'] = {
            #         'best_config': best_config.name,
            #         'num_matches': len(best_matches),
            #         'matches': best_matches[:10]  # Top 10 for brevity
            #     }
            # Method 2: Multi-configuration
        if 'multi_config' in methods:
            configs_results = self.multi_config.find_optimal_configurations(values1, values2)

            # Store ALL configurations and their matches
            all_configs_output = []
            for config, matches in configs_results:
                all_configs_output.append({
                    'config_name': config.name,
                    'threshold': config.threshold,
                    'num_matches': len(matches),
                    'matches': matches[:20]  # limit to top 20 for readability
                })

            results['multi_config'] = all_configs_output

        # Method 3: Distribution-based
        if 'distribution' in methods:
            similar = self.dist_matcher.are_columns_similar_by_distribution(col1, col2)
            results['distribution'] = {
                'similar': similar
            }
        
        return results
    
    def _apply_transformation_matches(self,
                                     values1: List[str],
                                     values2: List[str],
                                     trans_chain: List[str]) -> List[Tuple[str, str]]:
        """Apply transformation chain to find matches"""
        matches = []
        target_set = set(values2)
        
        for v1 in values1:
            transformed = self.transformer.apply_transformation_chain(v1, trans_chain)
            if transformed in target_set:
                matches.append((v1, transformed))
        
        return matches


# Example usage
# if __name__ == "__main__":
#     print("=== Transformation Discovery ===")
    
#     # Example 1: Abbreviation pattern
#     source = ["United States of America", "United Kingdom", "United Arab Emirates"]
#     target = ["USA", "UK", "UAE"]
    
#     trans_chain = ValueTransformation.discover_transformation(source, target)
#     print(f"Transformation chain: {trans_chain}")
    
#     # Example 2: Multi-configuration matching
#     print("\n=== Multi-Configuration Matching ===")
    
#     values1 = ["Manhattan", "Queens", "Brooklyn"]
#     values2 = ["MANHATTAN", "QUEENS ", "Brklyn"]
    
#     matcher = MultiConfigurationMatcher(min_precision=0.5)
#     configs = matcher.find_optimal_configurations(values1, values2)
    
#     for config, matches in configs[:3]:
#         print(f"\nConfig: {config.name}")
#         print(f"  Matches found: {len(matches)}")
#         for v1, v2, score in matches[:3]:
#             print(f"    {v1} <-> {v2} ({score:.3f})")
    
#     # Example 3: Distribution-based
#     print("\n=== Distribution-Based Matching ===")
    
#     col1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     col2 = pd.Series([1.1, 2.2, 3.1, 4.2, 5.0, 6.1, 7.2, 8.0, 9.1, 10.2])
    
#     similar = DistributionBasedMatching.are_columns_similar_by_distribution(col1, col2)
#     print(f"Columns similar by distribution: {similar}")
    
#     # Example 4: Hybrid matching
#     print("\n=== Hybrid Value Matching ===")
    
#     df1 = pd.DataFrame({'borough': ['Manhattan', 'Queens', 'Brooklyn']})
#     df2 = pd.DataFrame({'boro': ['Manh', 'Que', 'Brkln']})
    
#     hybrid = HybridValueMatcher()
#     results = hybrid.find_matches(df1['borough'], df2['boro'])
    
#     print("\nHybrid Results:")
#     for method, result in results.items():
#         print(f"  {method}: {result}")
if __name__ == "__main__":
    import argparse
    import sys

    # -------------------------
    # Parse CLI arguments
    # -------------------------
    parser = argparse.ArgumentParser(description="Hybrid value matcher")

    parser.add_argument("--left", required=True, help="Path to left CSV file")
    parser.add_argument("--right", required=True, help="Path to right CSV file")
    parser.add_argument("--left_col", required=True, help="Column name in left CSV")
    parser.add_argument("--right_col", required=True, help="Column name in right CSV")

    args = parser.parse_args()

    # -------------------------
    # Load CSVs
    # -------------------------
    left_df = pd.read_csv(args.left)
    right_df = pd.read_csv(args.right)

    if args.left_col not in left_df.columns:
        print(f"Column {args.left_col} not found in left CSV.")
        sys.exit(1)

    if args.right_col not in right_df.columns:
        print(f"Column {args.right_col} not found in right CSV.")
        sys.exit(1)

    col_left = left_df[args.left_col].astype(str)
    col_right = right_df[args.right_col].astype(str)

    # -------------------------
    # Initialize matcher
    # -------------------------
    hybrid = HybridValueMatcher()

    # -------------------------
    # Run hybrid matching
    # -------------------------
    results = hybrid.find_matches(col_left, col_right)

    # -------------------------
    # Prepare output file
    # -------------------------
    with open("output.txt", "w", encoding="utf-8") as f:

        f.write("=== HYBRID VALUE MATCHING REPORT ===\n\n")
        f.write(f"Left CSV: {args.left}\n")
        f.write(f"Right CSV: {args.right}\n")
        f.write(f"Column Left: {args.left_col}\n")
        f.write(f"Column Right: {args.right_col}\n\n")

        # ----------------------------------------------------
        # 1. TRANSFORMATION-BASED MATCHING RESULTS
        # ----------------------------------------------------
        f.write("=== Transformation-Based Matching ===\n")

        trans_result = results.get("transformation", {})

        if trans_result.get("found"):
            f.write("Transformation chain found: " +
                    " -> ".join(trans_result["chain"]) + "\n")
            f.write(f"Matches found: {len(trans_result['matches'])}\n\n")

            for v1, v2 in trans_result["matches"][:20]:
                f.write(f"  {v1} -> {v2}\n")

        else:
            f.write("No valid transformation chain discovered.\n\n")

        # ----------------------------------------------------
        # 2. MULTI-CONFIGURATION MATCHING
        # ----------------------------------------------------
        # f.write("\n=== Multi-Configuration Fuzzy Matching ===\n")

        # mc_result = results.get("multi_config", {})

        # if mc_result:
        #     f.write(f"Best Config: {mc_result.get('best_config')}\n")
        #     f.write(f"Number of Matches: {mc_result.get('num_matches')}\n\n")

        #     for v1, v2, score in mc_result.get("matches", [])[:20]:
        #         f.write(f"  {v1} <-> {v2} (score={score:.3f})\n")

        # else:
        #     f.write("No multi-configuration match results.\n\n")
        # 2. MULTI-CONFIGURATION MATCHING (ALL CONFIGS)
        f.write("\n=== Multi-Configuration Fuzzy Matching (All Configurations) ===\n")

        mc_results = results.get("multi_config", [])

        if not mc_results:
            f.write("No multi-configuration match results.\n")
        else:
            for cfg in mc_results:
                f.write(f"\n--- Configuration: {cfg['config_name']} ---\n")
                f.write(f"Threshold: {cfg['threshold']}\n")
                f.write(f"Matches Found: {cfg['num_matches']}\n\n")

                for v1, v2, score in cfg['matches']:
                    f.write(f"  {v1} <-> {v2} (score={score:.3f})\n")

        # ----------------------------------------------------
        # 3. DISTRIBUTION-BASED MATCHING
        # ----------------------------------------------------
        f.write("\n=== Distribution-Based Matching ===\n")

        dist_result = results.get("distribution", {})
        f.write(f"Columns similar by distribution: {dist_result.get('similar')}\n")

    print("\nResults written to output.txt")
