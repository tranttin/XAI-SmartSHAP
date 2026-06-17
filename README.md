# XAI-SmartSHAP

> Efficient and Faithful Sentence-Level Explainability for Scientific Document Recommendation

XAI-SmartSHAP is a research framework for evaluating explainability methods in scientific document recommendation systems. The project focuses on **sentence-level attribution**, helping identify which sentences within a paper abstract contribute most to its relevance score for a given query.

The framework provides a unified benchmark for comparing **SmartSHAP** against widely used explainability methods including KernelSHAP, LIME, Leave-One-Out (LOO), Banzhaf, and RISE.

---

# Research Contributions

This project introduces **SmartSHAP**, a lightweight and efficient explainability framework designed for document recommendation tasks.

Key contributions include:

### 1. Structured Coalition Sampling

Instead of relying on expensive random coalition generation used by traditional KernelSHAP, SmartSHAP employs structured sampling strategies that prioritize informative sentence combinations.

Benefits:

- Reduced computational cost
- Better sample efficiency
- Improved scalability for long abstracts

---

### 2. Query-Aware Sentence Selection

For long documents, SmartSHAP-Sentence performs query-guided sentence screening before explanation generation.

Benefits:

- Fewer explanation features
- Faster inference
- Better focus on relevant content

---

### 3. SHAP + Leave-One-Out Calibration

SmartSHAP combines cooperative game-theoretic attribution with direct removal-based importance estimation.

Benefits:

- Improved faithfulness
- More robust sentence ranking
- Better alignment with recommendation behavior

---

# Project Pipeline

```text
Dataset
   │
   ▼
Data Loader
   │
   ▼
Recommendation Model
   │
   ▼
Explanation Method
   │
   ▼
Sentence Attribution
   │
   ▼
XAI Evaluation
```

Detailed workflow:

```text
arXiv / S2ORC / Hartzbyte
            │
            ▼
         loader.py
            │
            ▼
    RecommenderModel
            │
            ▼
      Similarity Score
            │
            ▼
      SmartSHAP / LIME
      LOO / Banzhaf
      RISE / KernelSHAP
            │
            ▼
      Sentence Importance
            │
            ▼
       XAIEvaluator
            │
            ▼
     Metrics & Results
```

---

# Repository Structure

```text
XAI-SmartSHAP/
│
├── main.py
├── models.py
├── loader.py
├── xai_evaluator.py
├── crawl.py
├── chunk.py
│
├── data/
│   ├── arxiv/
│   ├── hartzbyte/
│   ├── s2orc/
│   └── source.txt
│
├── results/
│   ├── run_*/
│   ├── tensorboard_logs/
│   └── experiment outputs
│
└── README.md
```

---

# Components

## main.py

Main experiment runner.

Responsibilities:

- Load datasets
- Sample papers
- Execute explanation methods
- Evaluate explanations
- Save metrics and logs

---

## models.py

Contains both the recommendation model and explainability methods.

### Recommendation Model

The framework uses:

- `BAAI/bge-small-en`
- `BAAI/bge-reranker-base`

to estimate semantic relevance between queries and scientific abstracts.

### Implemented Explainers

#### Proposed Methods

- SmartSHAP
- SmartSHAP-Sentence
- Adaptive SmartSHAP-LOO
- Screener SmartSHAP

#### Baselines

- KernelSHAP
- LIME
- Leave-One-Out (LOO)
- Banzhaf
- RISE
- Random

---

## loader.py

Dataset loading and preprocessing.

Features:

- Dataset loading
- Abstract cleaning
- Sentence segmentation
- Automatic paper ID generation

---

## xai_evaluator.py

Evaluation module for explanation quality.

Implemented metrics:

### Faithfulness

- Comprehensiveness
- Sufficiency

### Agreement

- Spearman Rank Correlation

### Error

- MAE
- RMSE

### Stability

- Multi-run consistency

---

## crawl.py

Utility for extracting Computer Science papers from the official arXiv metadata dump.

Output:

```text
data/arxiv/papers.csv
```

---

## chunk.py

Optional preprocessing pipeline.

Capabilities:

- Sentence segmentation
- GPT-based sentence refinement
- Abstract enrichment

Output:

```text
papers_enriched.csv
```

---

# Datasets

The framework currently supports multiple scientific document collections.

```text
data/
├── arxiv/
├── hartzbyte/
└── s2orc/
```

Each dataset should contain a CSV file:

```text
data/<dataset_name>/papers.csv
```

Required columns:

```csv
title,abstract
```

Recommended columns:

```csv
title,abstract,categories,update_date,paper_id
```

---

# Results

All experiment outputs are stored in:

```text
results/
```

Example:

```text
results/
├── run_001/
├── run_002/
├── smartshap_vs_lime/
├── smartshap_vs_kernelshap/
└── tensorboard_logs/
```

Stored artifacts may include:

- Evaluation metrics
- Runtime statistics
- TensorBoard logs
- Experimental summaries
- Comparison tables

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/XAI-SmartSHAP.git
cd XAI-SmartSHAP
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Core packages:

```text
torch
sentence-transformers
transformers
shap
lime
numpy
pandas
scipy
scikit-learn
tensorboard
```

---

# Running Experiments

## SmartSHAP

```bash
python main.py \
  --dataset arxiv \
  --mode smartshap \
  --samples 50
```

---

## SmartSHAP-Sentence

```bash
python main.py \
  --dataset arxiv \
  --mode smartshap_sentence \
  --samples 50
```

---

## LIME

```bash
python main.py \
  --dataset arxiv \
  --mode lime
```

---

## KernelSHAP Baseline

```bash
python main.py \
  --dataset arxiv \
  --mode baseline_kernel
```

---

## Run All Methods

```bash
python main.py \
  --dataset arxiv \
  --mode all \
  --samples 50
```

---

# Evaluation Metrics

The framework evaluates explanations using multiple complementary metrics.

| Metric | Description |
|----------|-------------|
| Comprehensiveness | Score reduction after removing top-k important sentences |
| Sufficiency | Score retained using only top-k sentences |
| Rank Agreement | Spearman correlation with baseline SHAP |
| MAE | Attribution error against baseline |
| RMSE | Attribution error against baseline |
| Stability | Consistency across multiple runs |
| Runtime | Execution time per explanation |

---

# Example Experiment

```bash
python main.py \
  --dataset arxiv \
  --mode smartshap \
  --samples 100 \
  --ns_smart 200 \
  --ns_base 1000
```

Output:

```text
Avg Comprehensiveness
Avg Sufficiency
Avg Rank Agreement
Avg MAE
Avg RMSE
Avg Stability
Avg Runtime
```

---

# TensorBoard Visualization

Launch TensorBoard:

```bash
tensorboard --logdir results/
```

Open:

```text
http://localhost:6006
```

to visualize metrics and experiment logs.

---

# Future Work

Potential extensions include:

- Large Language Model explainers
- Cross-encoder attribution methods
- Multi-document recommendation
- Human-centered explanation evaluation
- Benchmarking on additional scientific datasets

---

# Citation

If you use this repository in academic research, please cite:

```bibtex
@software{xai_smartshap,
  title={XAI-SmartSHAP: Efficient Sentence-Level Explainability for Scientific Document Recommendation},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repository}
}
```

---

# License

MIT License.

---
