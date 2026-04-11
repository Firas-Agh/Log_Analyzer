import os
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from chunking.chunk_code import chunk_codebase
from chunking.chunk_logs import chunk_logs

def embed_chunks(chunks, embed_model):
    texts = [c["content"] for c in chunks]

    vectors = embed_model.encode(texts, show_progress_bar=True)
    return np.array(vectors).astype("float32")

def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def search(query, index, chunks, top_k=50):
    # 1. embed query
    query_vec = embed_model.encode([query]).astype("float32")

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

    chunks = code_chunks + log_chunks

    embed_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

    vectors = embed_chunks(chunks, embed_model)

    faiss_indexes = build_faiss_index(vectors)

    results = search("why are some requests failing?", faiss_indexes, chunks)
