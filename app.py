from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from catch import ClusterSemanticCatch

app = FastAPI()
model = SentenceTransformer("models/minilm")

index=faiss.read_index("data/news_index.faiss")

with open("data/doc_mapping.plk","rb") as f:
    doc=pickle.load(f)
    
with open("data/gmm_model.pkl","rb") as f:
    gmm=pickle.load(f)
    
    
cache=ClusterSemanticCatch()


@app.get("/")
def home():
    return {"message": "API working"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/query")
def query(body: dict):

    query = body["query"]

    query_emb = model.encode([query]).astype("float32")

    cluster_probs = gmm.predict_proba(query_emb)
    cluster_id = int(cluster_probs.argmax()) 
    cached = cache.incatch(query_emb, cluster_id)

    if cached:

        entry, score = cached

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": score,
            "result": entry["result"],
            "dominant_cluster": cluster_id
        }
    d, l = index.search(query_emb, 5)

    result = doc[l[0][0]]

    cache.add(query, query_emb, result, cluster_id)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": float(d[0][0]),
        "result": result,
        "dominant_cluster": cluster_id
    }
    
    
@app.get('/cache/stats')
def status():
        return cache.catch_status()
        
        
@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {
        "message": "Cache flushed successfully"
    }
