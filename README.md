# 🔍 RAG Evaluation Metrics with DeepEval

A comprehensive Jupyter notebook that walks through building a **Retrieval-Augmented Generation (RAG)** pipeline from scratch and rigorously evaluating it using **DeepEval** — covering both retriever-level and generator-level metrics.

---

## 📌 Overview

RAG systems are only as good as what they retrieve and what they generate. This notebook bridges the gap between building a RAG pipeline and **actually knowing if it works well** — using structured, LLM-judged evaluation metrics.

You'll go from raw CSV data → vector store → RAG chain → full evaluation suite in one end-to-end flow.

---

## 🧱 Tech Stack

| Tool | Purpose |
|------|---------|
| `LangChain` | RAG pipeline orchestration |
| `OpenAI (GPT-4o-mini)` | LLM for answer generation |
| `OpenAI Embeddings (text-embedding-3-small)` | Document embedding |
| `ChromaDB` | Vector store for semantic retrieval |
| `DeepEval` | RAG evaluation framework |
| `GPT-4o` | LLM judge for all metrics |
| `Pandas` | Data loading and preprocessing |

---

## 🗂️ Notebook Structure

### 1. 🔧 Setup & Dependencies
Install all required libraries:
```
langchain, langchain-openai, langchain-community,
langchain-chroma, deepeval, dill
```
Configure your OpenAI API key securely using `getpass`.

---

### 2. 📂 Data Loading & Preprocessing
- Load a CSV dataset (`rag_eval_docs.csv`) containing document contexts and titles.
- Convert records into LangChain `Document` objects with proper `metadata` (title, id) and `page_content`.
- Skip chunking since the dataset is compact enough to embed as-is.

---

### 3. 🧠 Embeddings & Vector Store (ChromaDB)
- Embed all documents using `text-embedding-3-small`.
- Store them in a **ChromaDB** collection with **cosine similarity** space (`hnsw:space: cosine`).
- Persist the vector DB to disk for reuse.
- Load the persisted DB in subsequent runs without re-embedding.

---

### 4. 🔎 Semantic Retrieval
- Use `similarity_score_threshold` retriever (`k=3`, `score_threshold=0.3`).
- Only documents with a similarity score ≥ 0.3 are returned.
- Tested with queries like `"What is AI?"` and `"What is Photosynthesis?"`.

---

### 5. 🤖 RAG Pipeline
Build a complete RAG chain using LangChain's LCEL (LangChain Expression Language):

```
User Query
   ↓
Similarity Retriever (ChromaDB)
   ↓
Format Retrieved Docs
   ↓
RAG Prompt Template
   ↓
GPT-4o-mini (Generator)
   ↓
Final Answer + Source Context
```

The prompt instructs the LLM to **only answer from retrieved context** and say "I don't know" if the answer isn't there — preventing hallucinations by design.

---

### 6. 📊 Evaluation Metrics (DeepEval)

All metrics use **GPT-4o as the LLM judge**, with a default threshold of `0.55`.

#### 🔵 Retriever Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Contextual Precision** | Are the retrieved chunks ranked correctly — relevant chunks at the top? |
| **Contextual Recall** | Does the retrieved context cover everything needed to answer the expected output? |
| **Contextual Relevancy** | Are all retrieved chunks actually relevant to the query? |

Each retriever metric is demonstrated with different `retrieval_context` setups — including deliberately noisy/irrelevant contexts — to show how scores degrade with bad retrieval.

---

#### 🟢 Generator Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Answer Relevancy** | Does the LLM's answer actually address the user's question? |
| **Faithfulness** | Is the answer grounded in the retrieved context (no made-up facts)? |
| **Hallucination** | Does the answer contradict the ground-truth context? (Lower = better) |
| **G-Eval (Custom LLM Judge)** | Chain-of-thought evaluation using custom scoring criteria |

---

#### 🟡 G-Eval — Custom Evaluation (LLM as a Judge)

G-Eval lets you define **your own evaluation criteria** using natural language steps. In this notebook, a custom `"RAG Fact Checker"` metric is built that evaluates:

1. Extract statements from the actual output
2. Check if statements answer the input question (penalize irrelevant ones)
3. Compare against the expected output (penalize missing/wrong facts)
4. Verify grounding in the retrieval context (penalize ungrounded claims)
5. Penalize any invented or factually nonsensical statements

---

### 7. 🌟 Synthetic Data Generation (Goldens)
- Use DeepEval's `Synthesizer` with `GPT-4o` to **auto-generate evaluation questions** from document contexts.
- `generate_goldens_from_contexts` creates `(question, expected_output, context)` triples — useful for building automated eval datasets without manual labeling.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- OpenAI API Key (with access to `gpt-4o-mini` and `gpt-4o`)

### Installation
```bash
pip install langchain==0.3.10
pip install langchain-openai==0.2.12
pip install langchain-community==0.3.11
pip install langchain-chroma==0.1.4
pip install deepeval
pip install dill
```

### Running the Notebook
1. Clone the repository
2. Launch Jupyter: `jupyter notebook`
3. Open `evalution_metrics_rag.ipynb`
4. Enter your OpenAI API key when prompted
5. Run all cells sequentially

> **Note:** You'll need the `rag_eval_docs.csv` dataset. Download it using the `gdown` cell in the notebook or provide your own CSV with `id`, `title`, and `context` columns.

---

## 📐 Evaluation Flow Summary

```
                    ┌─────────────┐
                    │  RAG Query  │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │     ChromaDB Retriever  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    GPT-4o-mini Answer   │
              └────────────┬────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │         DeepEval Evaluation        │
         │                                   │
         │  RETRIEVER       GENERATOR         │
         │  ─────────       ─────────         │
         │  Precision       Answer Relevancy  │
         │  Recall          Faithfulness      │
         │  Relevancy       Hallucination     │
         │                  G-Eval (Custom)   │
         └───────────────────────────────────┘
```

---

## 📁 File Structure

```
├── evalution_metrics_rag.ipynb   # Main notebook
├── rag_eval_docs.csv             # Evaluation dataset (download via gdown)
└── my_db/                        # Persisted ChromaDB vector store (auto-created)
```

---

## 💡 Key Concepts Demonstrated

- **Why retrieval quality matters** — shown by injecting noisy/irrelevant contexts and watching metric scores drop
- **Difference between Recall vs Precision** — using carefully crafted `new_context` examples
- **Hallucination detection** — using ground-truth context vs model output comparison
- **Custom evaluation** — G-Eval with chain-of-thought scoring steps
- **Synthetic test data** — auto-generating QA pairs for scalable evaluation

---

## 📚 References

- [DeepEval Documentation](https://docs.confident-ai.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

---


## 📄 License

This project is for educational purposes. Feel free to use and adapt it for your own learning.

