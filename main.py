def main():
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from pipeline import RAGPipeline
    CONTEXT_DIR = "data/context.txt"
    query = "What are some of the skills taught in the Trail Patrol Training course?"
    pipeline = RAGPipeline()
    if not pipeline.embed_handler.load_index():
        print("Index not found, ingesting data...")
        if os.path.exists(CONTEXT_DIR):
            pipeline.ingest(CONTEXT_DIR)
        else:
            print(f"Error: Context file {CONTEXT_DIR} not found.")
            return
    print(f"Query: {query}\n")
    print("Response:")
    completion = pipeline.chat(query, k=10, j=3, stream=True)
    if completion:
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()

if __name__ == "__main__":
    main()