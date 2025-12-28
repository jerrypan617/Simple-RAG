import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import CrossEncoder
import torch

class RerankerHandler:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5", cache_folder="models/cross_encoder"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device, cache_folder=cache_folder)

    def rerank(self, candidates, query, top_k=3):
        """
        candidates: list of (text, distance).
        query: str
        """
        if isinstance(query, (list, tuple)) and len(query) == 1 and isinstance(query[0], str):
            query_text = query[0]
        elif isinstance(query, str):
            query_text = query
        else:
            raise ValueError("query must be a text string.")
        valid_candidates = []
        for item in candidates:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            text = item[0]
            if text:
                valid_candidates.append(item)
        if not valid_candidates:
            return []
        pairs = [[query_text, item[0]] for item in valid_candidates]
        scores = self.model.predict(pairs)
        combined = []
        for (text, dist), score in zip(valid_candidates, scores):
            combined.append((text, float(score), float(dist)))
        combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
        return combined_sorted[:top_k]