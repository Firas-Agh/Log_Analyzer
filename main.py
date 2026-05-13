import math
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np
import requests
from huggingface_hub import InferenceClient

from chunking.cap_chunks import cap_chunks_by_tokens
from chunking.chunk_code import chunk_codebase
from chunking.chunk_logs import chunk_logs

def embed_chunks(chunks, batch_size, max_workers):

    divided_chunks = split_into_thread_chunks(chunks, max_workers)

    logging.info(f"divided chunks list into {len(divided_chunks)} lists")

    all_vectors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(
                process_thread_chunk,
                thread_chunk,
                batch_size,
                idx
            )
            for idx, thread_chunk in enumerate(divided_chunks)
        ]

        for future in as_completed(futures):
            vectors = future.result()
            all_vectors.extend(vectors)

    return np.array(vectors).astype("float32")

def split_into_thread_chunks(data, num_threads):
    """
    Split the big list into N nearly equal parts.
    """
    chunk_size = math.ceil(len(data) / num_threads)

    return [
        data[i:i + chunk_size]
        for i in range(0, len(data), chunk_size)
    ]

def initialize_client():
    client = InferenceClient(
        provider="hf-inference",
        api_key=os.environ["embeddinggemma-300m TOKEN"],
    )

    return client

def process_thread_chunk(thread_chunks, batch_size, list_index):
    logging.info(f"processing thread chunk #{list_index}")
    client = initialize_client()

    vectors = []

    for i in range(0, len(thread_chunks), batch_size):
        batch = thread_chunks[i:i + batch_size]

        vectors_partial = embed_text(client, batch)
        vectors.extend(vectors_partial)

    return vectors

def embed_text(client, batch):
    result = client.feature_extraction(
        batch,
        model="google/embeddinggemma-300m"
    )
    return result



def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

# def search(query, index, chunks, top_k=50):
#     # 1. embed query
#     query_vec = embed_text(query)

#     # 2. search FAISS
#     distances, indices = index.search(query_vec, top_k)

#     # 3. map results back to chunks
#     results = [chunks[i] for i in indices[0]]

#     return results

if __name__ == "__main__":
    path = "log files/almatarhotels_run_prod_79.log"
    code_repo_dir="C:/Users/user/Desktop/SEERA/github new repo's/crawla_almatarhotels"

    logging.basicConfig(level=logging.INFO)

    size = os.path.getsize(path)
    size_mb = size/1000000

    log_chunks = chunk_logs(path, size_mb)


    code_chunks = chunk_codebase(code_repo_dir)

    chunks = cap_chunks_by_tokens(code_chunks + log_chunks)

    batch_size = 1000
    threads = 30

    logging.info(f"embedding chunks ...")
    embedding_start = time.time()
    vectors = embed_chunks(chunks, batch_size, threads)
    embedding_end = time.time()
    logging.info(f"took {embedding_end - embedding_start} to embed all chunks")

    faiss_indexes = build_faiss_index(vectors)

    # results = search("why are some requests failing?", faiss_indexes, chunks)
