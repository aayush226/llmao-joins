# LLMao-Joins: Semantic Joins and Value Matching Operators using Graph-Based Reasoning

### Team: LLMao Joins
**Members:**  
- Aayush Shah (ajs10171)  
- Bineet Singh Chadha (bsc6177)  
- Kai Kang (ck4193)

---

## 🧩 Project Overview

When two tables store the same information in different ways, traditional joins fail.  
For example, “USA” and “United States of America” refer to the same country but would not match in a normal SQL join.

**LLMAO-JOINS** is a hybrid *semantic join operator* that combines:
- **String and embedding-based similarity**
- **Graph-based reasoning** using Neo4j
- **Selective use of LLMs** for semantic disambiguation

The goal is to achieve *accurate, efficient, and explainable joins* that handle both spelling-level and meaning-level differences while keeping computational and token cost low:contentReference[oaicite:1]{index=1}.

---

## ⚙️ System Architecture

Our system runs as a multi-stage pipeline with reproducible modules:

1. **Normalization**  
   Clean and standardize text — lowercase, remove punctuation, expand abbreviations.

2. **Candidate Generation**  
   Use n-gram overlap, edit distance, and short-text embeddings (via FAISS or cosine similarity) to generate likely match pairs.

3. **Graph Construction (Neo4j)**  
   Build a *value-alias graph* where each unique value is a node and edges encode different types of similarity:
   - `RULE_MATCH`: heuristic or abbreviation rule  
   - `SIM_MATCH`: numeric similarity or embedding match  
   - `LLM_VERIFIED`: confirmed by GPT model  

4. **Graph Clustering**  
   Apply label propagation or node similarity to find clusters of equivalent values, enabling transitive inference (A≈B, B≈C ⇒ A≈C).

5. **Scoring and Filtering**  
   Compute a weighted confidence score combining multiple similarity metrics. Uncertain pairs are sent to an LLM gate.

6. **LLM Verification (Budgeted)**  
   Only ambiguous pairs are checked using an OpenAI GPT model, keeping token cost predictable.

7. **Join Generation & Explanation**  
   Produce final joins with explanations for why values matched (e.g., “matched due to embedding similarity and graph connectivity”).

---

## 🧠 Why the Graph Matters

Traditional fuzzy joins treat each pair independently.  
Our **graph-based approach** adds:

- **Transitive reasoning** — captures indirect matches (e.g., “U.S.” → “USA” → “United States of America”)  
- **Knowledge retention** — reuses previous discovered matches across datasets  
- **Explainability** — the graph shows exactly *why* two values matched  
- **Scalability** — incremental graph updates instead of recomputing all pairwise similarities

---

## 🧪 Datasets and Evaluation

We evaluate LLMAO-JOINS on benchmark datasets:

- **Auto-FuzzyJoin Benchmark:** Measures precision and recall against ground-truth mappings.  
- **Freyja Dataset:** Tests efficiency and scalability of joins on large, noisy data.

**Metrics:**
- Precision, Recall, F1-score
- Join coverage
- Pair reduction rate
- Execution time
- Token cost vs accuracy tradeoff for the LLM module

---

## 🧱 Repository Structure
```python
llmao-joins/
│
├── llmao_joins/
│ ├── pipeline.py # Main entry point & orchestrator
│ ├── config.py # Configurations & default parameters
│ ├── io_and_normalization.py # Text cleaning and abbreviation expansion
│ ├── data_preprocessing.py # Tokenization and similarity computation
│ ├── candidate_generation.py # Heuristic and embedding-based candidate search
│ ├── scoring_and_llm.py # Scoring model and LLM verification logic
│ ├── graph_and_clusters.py # Neo4j integration and label propagation
│ ├── value_matching.py # Additional rule-based similarity logic
│ └── llm_client.py # Interface for GPT API calls
│
├── benchmark_runner.py # For reproducible evaluations
├── left_dummy.csv / right_dummy.csv # Example input data
├── requirements.txt
└── README.md

```


---

## 💻 Installation & Setup

### 1. Clone and Navigate
```bash
git clone https://github.com/aayush226/llmao-joins.git
cd llmao-joins
```

### 2. Set Up Environment
```bash
python3 -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a .env file with:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yourpassword
LLM_API_KEY=your_openai_api_key  # optional
```

### 4. Running the Pipeline
You can run the entire semantic join pipeline using the CLI:
```bash
python -m llmao_joins.pipeline \
  --left_csv left.csv \
  --right_csv right.csv \
  --left_col title \
  --right_col title \
  --output_dir outputs
```

### Optional Flags

| Flag | Description |
|------|--------------|
| `--embed_model_name` | Sentence-transformer model (default: `all-MiniLM-L6-v2`) |
| `--neo4j_uri`, `--neo4j_user`, `--neo4j_password` | Override Neo4j credentials |
| `--llm_api_key` | Use a custom OpenAI API key |
| `--abbrevation_master` | Custom abbreviation mapping CSV |

---

## 📊 Outputs

All outputs are written to the folder specified by `--output_dir` (default: `outputs/`):

| File | Description |
|------|--------------|
| `matched_pairs.csv` | High-confidence matched pairs with similarity scores |
| `semantic_join.csv` | Final joined output |
| `synonym_clusters.csv` | Cluster assignments from Neo4j |
| `metrics.json` | Runtime, pair counts, and evaluation metrics |

**Example log:**

```bash
[LLMAO-JOINS] Finished pipeline. Outputs written to outputs
[LLMAO-JOINS] Matched pairs: 234, joined rows: 122
[LLM GATE] LLM calls: 47, Tokens: 6210, Cost: $0.93
```

## 🔁 Reproducibility

This project was designed to ensure **complete reproducibility** of results:

1. **Freeze dependencies** via `requirements.txt`.  
2. **Record configurations** from `llmao_joins/config.py`.  
3. **Persist graph knowledge** using a Neo4j export/dump.  
4. **Use the same dataset versions** (`left.csv`, `right.csv`).  
5. **Document LLM version** (e.g., `gpt-4o-mini` or similar).  

Rerunning the same command under identical conditions produces the same results.

---

## 📈 Project Milestones

| Milestone | Description |
|------------|-------------|
| M1 | Baseline fuzzy join setup |
| M2 | Candidate generation using string techniques |
| M3 | Graph construction in Neo4j + clustering |
| M4 | Scoring and selective LLM verification |
| M5 | Final evaluation and reproducibility validation |

---

## 📚 References

- *Auto-FuzzyJoin* (SIGMOD 2021)  
- *Valentine Framework* (ICDE 2021)  
- *Zhang et al., Distribution-based Matching* (SIGMOD 2011)

---

## 👥 Authors and Contact

Developed by **Aayush Shah, Bineet Singh Chadha, and Kai Kang**  
**Team:** LLMao Joins  
**Course:** Big Data, Fall 2025  
**Repository:** [https://github.com/aayush226/llmao-joins](https://github.com/aayush226/llmao-joins)
