#!/usr/bin/env python3
"""
ContextFit Quickstart Example.

Demonstrates:
1. Creating a knowledge base
2. Ingesting text
3. Querying (staying in token space)
4. Accessing input_ids directly
"""

import tempfile
from pathlib import Path

from contextfit import RetrievalEngine, Tokenizer


def main():
    # Sample knowledge
    documents = [
        """
        Python async programming uses coroutines and the async/await syntax.
        The asyncio module provides the event loop that drives async execution.
        Coroutines are defined with 'async def' and called with 'await'.
        This allows concurrent execution without threads.
        """,
        """
        JavaScript also has async/await, inspired by Python's approach.
        Node.js uses a single-threaded event loop for non-blocking I/O.
        Promises in JavaScript are similar to Python's Future objects.
        Both languages handle concurrency without traditional multithreading.
        """,
        """
        Rust's async model is different - it's zero-cost abstraction.
        Futures in Rust are lazy and need an executor to run.
        Tokio is the most popular async runtime for Rust.
        Unlike Python, Rust async has no garbage collector overhead.
        """,
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create engine
        print("Creating ContextFit knowledge base...")
        engine = RetrievalEngine.create(Path(tmpdir) / "kb")
        
        # Ingest documents
        print("Ingesting documents...")
        for i, doc in enumerate(documents):
            chunks = engine.ingest_text(doc, chunk_size=128, overlap=16)
            print(f"  Document {i+1}: {len(chunks)} chunks")
        
        # Show stats
        stats = engine.stats()
        print(f"\nKnowledge base stats:")
        print(f"  Chunks: {stats['chunks']}")
        print(f"  Storage: {stats['storage']['data_size_bytes']} bytes (compressed)")
        print(f"  Vocab: {stats['index']['vocab_size']} unique tokens")
        
        # Query
        print("\n" + "="*50)
        query = "How does Python handle async programming?"
        print(f"Query: {query}")
        
        result = engine.query(query, top_k=3, method="bm25")
        
        print(f"\nQuery tokenized to {len(result.query_tokens)} tokens")
        print(f"Retrieved {len(result.chunks)} chunks, {len(result.input_ids)} total tokens")
        
        # Show results
        print("\n--- Retrieved Content ---")
        for i, (chunk, score) in enumerate(zip(result.chunks, result.scores)):
            print(f"\n[Chunk {i+1}] Score: {score:.4f}")
            # Decode just for display
            text = engine.tokenizer.decode(chunk.tokens)
            print(text.strip()[:200] + "..." if len(text) > 200 else text.strip())
        
        # The key insight: input_ids can go directly to an LLM
        print("\n--- Input IDs (ready for LLM) ---")
        print(f"Shape: {result.input_ids.shape}")
        print(f"First 20 token IDs: {result.input_ids[:20].tolist()}")

        # SID generator path
        engine.train_learned_sid_generator()
        sid_result = engine.query(query, top_k=2, method="sid")
        print("\n--- SID Generator ---")
        if sid_result.sid_predictions:
            best = sid_result.sid_predictions[0]
            print(f"Best SID prefix: {list(best.prefix)}")
            print(f"Score: {best.score:.4f}, depth: {best.depth}, support: {best.support}")
        print(f"SID retrieval returned {len(sid_result.chunks)} chunks")
        
        # Demonstrate: no detokenization needed until final output
        print("\n✓ All operations stayed in token space until this display step!")


if __name__ == "__main__":
    main()
