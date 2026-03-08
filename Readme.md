# 🚀 Semantic Search System with Cluster-Aware Cache

A lightweight **semantic search system** built on the **20 Newsgroups dataset (~20K documents)**.

This project implements:

- 🔎 Vector embeddings
- 🧠 Fuzzy clustering
- ⚡ Semantic caching
- 🌐 FastAPI service
- 🐳 Docker containerization

The system detects **semantically similar queries** and avoids recomputing results using a custom semantic cache.

---

## 🧠 Problem Statement

Traditional caches rely on **exact string matching**.

Example:

Query 1
```
space shuttle launch
```

Query 2
```
NASA rocket launch
```

Even though these queries mean the same thing, traditional caches treat them as different queries.

This system solves the problem using **semantic similarity**.

---

## 🏗 System Architecture

```
User Query
    │
    ▼
SentenceTransformer Embedding
    │
    ▼
Gaussian Mixture Clustering
    │
    ▼
Cluster-Aware Semantic Cache
    │
 ┌──┴──────────────┐
 │                 │
Cache Hit       Cache Miss
 │                 │
 ▼                 ▼
Return Result   FAISS Search
                    │
                    ▼
              Store in Cache
```

---

## 📚 Dataset

Dataset used:

**20 Newsgroups Dataset**

- ~20,000 discussion posts
- 20 topic categories
- Common NLP benchmark dataset

Dataset source:

https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

---

## 🧹 Data Cleaning

The raw dataset contains significant noise including:

- email headers
- quoted replies
- URLs
- signatures
- formatting artifacts

Cleaning steps performed:

- removed email headers
- removed quoted replies
- removed URLs
- removed very short documents

This ensures embeddings capture **semantic meaning rather than metadata**.

---

## 🧠 Embedding Model

Model used:

```
sentence-transformers/all-MiniLM-L6-v2
```

Reasons:

- 384 dimensional embeddings
- fast inference
- strong semantic similarity performance
- lightweight for real-time systems

---

## 🧩 Fuzzy Clustering

Clustering is performed using **Gaussian Mixture Models (GMM)**.

Unlike hard clustering algorithms such as K-Means, GMM produces **probabilistic cluster membership**.

Example:

```
Document A
Cluster 3 → 0.62
Cluster 7 → 0.28
Cluster 2 → 0.10
```

This reflects the fact that documents often belong to **multiple topics simultaneously**.

---

## ⚡ Semantic Cache

Traditional cache:

```
query string → exact match
```

Semantic cache:

```
embedding similarity → cache reuse
```

Similarity check:

```
cosine_similarity(query_embedding, cached_query_embedding)
```

If similarity exceeds threshold:

```
similarity > threshold
```

the cached result is reused.

---

## 🎛 Tunable Parameter

The semantic cache is controlled by a **similarity threshold**.

```
similarity_threshold
```

Example behavior:

| Threshold | Behavior |
|----------|----------|
0.70 | many cache hits but lower accuracy |
0.85 | balanced |
0.95 | fewer hits but higher precision |

---

## 🌐 FastAPI Endpoints

### POST /query

Input

```json
{
 "query": "space shuttle launch"
}
```

Response

```json
{
 "query": "...",
 "cache_hit": true,
 "matched_query": "...",
 "similarity_score": 0.91,
 "result": "...",
 "dominant_cluster": 3
}
```

---

### GET /cache/stats

Returns cache statistics.

```json
{
 "total_entries": 42,
 "hit_count": 17,
 "miss_count": 25,
 "hit_rate": 0.405
}
```

---

### DELETE /cache

Flushes the cache and resets statistics.

---

## ⚙ Running the Project

### Create virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```
pip install -r requirements.txt
```

### Start API server

```
uvicorn app:app --reload
```

API documentation

```
http://127.0.0.1:8000/docs
```

---

## 🐳 Docker Setup

Build Docker image

```
docker build -t trademark-ai-search .
```

Run container

```
docker run -p 8000:8000 trademark-ai-search
```

API available at

```
http://localhost:8000/docs
```

---

## 📂 Project Structure

```
TRADEMARK/
│
├── app.py
├── catch.py
├── Dockerfile
├── requirements.txt
├── README.md
│
├── data/
│   ├── gmm_model.pkl
│   ├── news_index.faiss
│   └── doc_mapping.plk
│
└──── data_process.ipynb
   
```

---

## 🚀 Future Improvements

Possible improvements:

- FAISS-based semantic cache lookup
- distributed caching
- dynamic cluster updates
- query batching

---

## 🎯 Conclusion

This project demonstrates how **vector embeddings, probabilistic clustering, and semantic caching** can be combined to build an efficient semantic search system.

By incorporating fuzzy clustering and semantic similarity, the system avoids redundant computation and scales significantly better than traditional caching approaches.