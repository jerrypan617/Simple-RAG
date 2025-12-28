import os

#OpenMP conflict on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import numpy as np
import faiss
import pickle

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class EmbeddingHandler:
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5", cache_folder="models/embedding"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device, cache_folder=cache_folder)
        self.index = None
        self.texts = []

    def embed_texts(self, texts, batch_size=32):
        embeddings_batches = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            batch_emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings_batches.append(batch_emb)
        if not embeddings_batches:
            return np.array([])
        return np.vstack(embeddings_batches)

    def embed_query(self, query):
        return self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)

    def build_index(self, texts, index_path="index/faiss.index", mapping_path="index/id2text.pkl", batch_size=32):
        self.texts = texts
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        emb = np.asarray(embeddings, dtype='float32')
        if emb.ndim != 2:
            raise ValueError("embeddings must be 2D array")
        N, D = emb.shape
        self.index = faiss.IndexFlatL2(D)
        self.index.add(emb)
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(mapping_path, "wb") as f:
            pickle.dump(texts, f)

    def load_index(self, index_path="index/faiss.index", mapping_path="index/id2text.pkl"):
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            return False
        self.index = faiss.read_index(index_path)
        with open(mapping_path, "rb") as f:
            self.texts = pickle.load(f)
        return True

    def search(self, query, k=5):
        if self.index is None:
            raise ValueError("Index not loaded or built.")
        query_emb = self.embed_query(query)
        q = np.asarray(query_emb, dtype='float32')
        if q.ndim == 1:
            q = q.reshape(1, -1)
        Dists, Ids = self.index.search(q, k)
        results = []
        for dist_row, id_row in zip(Dists, Ids):
            row = []
            for idx, dist in zip(id_row, dist_row):
                if idx < 0 or idx >= len(self.texts):
                    continue
                row.append((self.texts[idx], float(dist)))
            results.append(row)
        return results
