from src import embedding, loader, reranker, generator
import os

class RAGPipeline:
    def __init__(self):
        self.embed_handler = embedding.EmbeddingHandler()
        self.rerank_handler = reranker.RerankerHandler()
        self.generator = generator.Generator()
    
    def ingest(self, text_dir):
        chunklist = loader.chunk(text_dir)
        self.embed_handler.build_index(chunklist)
        
    def retrieve(self, query, k=10, j=3):
        if self.embed_handler.index is None:
             if not self.embed_handler.load_index():
                 raise ValueError("Index not found. Please run ingest() first.")
        results = self.embed_handler.search(query, k=k)
        if results:
            candidates = results[0]
        else:
            candidates = []
        res = self.rerank_handler.rerank(candidates, query, top_k=j)
        return "\n".join([text for text, _, _ in res])

    def chat(self, query, k=10, j=3, stream=True):
        context = self.retrieve(query, k=k, j=j)
        print("Context:\n", context)
        print("\n")
        return self.generator.chat(query, context, stream=stream)

_pipeline = None

def retrieve(query, text_dir, k=10, j=3):
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    if not _pipeline.embed_handler.load_index():
        print("Index not found, building index...")
        _pipeline.ingest(text_dir)
    return _pipeline.retrieve(query, k=k, j=j)