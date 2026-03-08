from sentence_transformers import SentenceTransformer


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ClusterSemanticCatch:
    def __init__(self,th=0.85):
        self.threshold=th
        self.cluster_catch={}
        
        self.hit_count=0
        self.miss_count=0
        
    def incatch(self,query_embedding,cluester_id):
        
        if cluester_id not in self.cluster_catch:
            self.miss_count+=1
            return None
        
        candidate=self.cluster_catch[cluester_id]
        
        best_match=0
        best_score=0
        
        for entry in self.cluster_catch[cluester_id]:
            
            sim=cosine_similarity(query_embedding,entry['embedding'])[0][0]
            
            if sim > best_score:
                best_score=float(sim)
                best_match=entry
                
        if best_score>self.threshold:
            
            self.hit_count+=1
            return best_match,best_score
        
        self.miss_count+=1
        return None
    
    def add(self,query,embedding,result,cluster):
        
        entry={
            "query":query,
            "embedding":embedding,
            "result":result
        }
        if cluster not in self.cluster_catch:
            self.cluster_catch[cluster]=[]
        self.cluster_catch[cluster].append(entry)
        
    def catch_status(self):
        print(self.cluster_catch)
        return { 
            "total_entries":len(self.cluster_catch),
            "hit_count":self.hit_count,
            "miss_count":self.miss_count,
            "hit_rate":round(self.hit_count/(self.hit_count+self.miss_count) if  (self.hit_count+self.miss_count)> 0 else 0,2) 
        }
        
    def clear(self):
        self.cluster_catch={}
        self.hit_count=0
        self.miss_count=0
    
        


