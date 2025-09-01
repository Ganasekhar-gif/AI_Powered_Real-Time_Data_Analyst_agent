import os, json
from typing import List, Dict, Any
import numpy as np
import faiss


class FaissIndex:
    def __init__(self, index_path: str, meta_path: str, emb):
        self.index_path = index_path
        self.meta_path = meta_path
        self.emb = emb
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self._load()


    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            vec = self.emb.encode(["init"]).astype("float32")
            d = vec.shape[1]
            self.index = faiss.IndexFlatIP(d) # cosine if normalized
            self.meta = []


    def add(self, docs: List[Dict[str, Any]]):
        texts = [d["text"] for d in docs]
        vecs = self.emb.encode(texts).astype("float32")
        self.index.add(vecs)
        self.meta.extend(docs)


    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)


    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if len(self.meta) == 0:
            return []
        vec = self.emb.encode([query]).astype("float32")
        D, I = self.index.search(vec, k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            m = self.meta[idx].copy()
            m["score"] = float(score)
            hits.append(m)
        return hits 
