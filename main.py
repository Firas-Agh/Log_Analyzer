import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import requests

from chunking.cap_chunks import cap_chunks_by_tokens
from chunking.chunk_code import chunk_codebase
from chunking.chunk_logs import chunk_logs

def embed_chunks(chunks):

    with ThreadPoolExecutor(max_workers=10) as ex:
        vectors = list(ex.map(embed_text, chunks))

    return np.array(vectors).astype("float32")

def embed_text(chunk):
    text = chunk if isinstance(chunk, str) else chunk.get("content", "")
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "mxbai-embed-large:latest",
            "prompt": text,
        },
    )

    return response.json()["embedding"]

def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def search(query, index, chunks, top_k=50):
    # 1. embed query
    query_vec = embed_text(query)

    # 2. search FAISS
    distances, indices = index.search(query_vec, top_k)

    # 3. map results back to chunks
    results = [chunks[i] for i in indices[0]]

    return results

if __name__ == "__main__":
    path = "log files/almatarhotels_run_prod_79.log"
    code_repo_dir="C:/Users/user/Desktop/SEERA/github new repo's/crawla_almatarhotels"

    logging.basicConfig(level=logging.INFO)

    size = os.path.getsize(path)
    size_mb = size/1000000

    log_chunks = chunk_logs(path, size_mb)

    code_chunks = chunk_codebase(code_repo_dir)

    chunks = cap_chunks_by_tokens(code_chunks + log_chunks)

    # embed_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

    logging.info(f"embedding chunks ...")
    embedding_start = time.time()
    vectors = embed_chunks(chunks)
    embedding_end = time.time()
    logging.info(f"took {embedding_end - embedding_start} to embed all chunks")

    faiss_indexes = build_faiss_index(vectors)

    results = search("why are some requests failing?", faiss_indexes, chunks)
