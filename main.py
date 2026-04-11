import os
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

from chunking.chunk_code import chunk_codebase
from chunking.chunk_logs import chunk_logs

import re

def embed_chunks(chunks):
    texts = []

    for c in chunks:
        # split oversized chunks
        parts = split_chunk(c)

        texts.extend(parts)

    with ThreadPoolExecutor(max_workers=10) as ex:
        vectors = list(ex.map(embed_text, texts))

    return np.array(vectors).astype("float32")

def embed_text(chunk):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "mxbai-embed-large:latest",
            "prompt": chunk.get("content","")
        }
    )

    return response.json()["embedding"]

def split_chunk(chunk, max_chars=500):
    """
    Splits a single chunk into smaller embedding-safe chunks.
    Works differently for logs vs code.
    """

    ctype = chunk.get("type")
    content = chunk.get("content", "")

    result = []

    # -------------------------
    # 1. LOG CHUNKING (line-based)
    # -------------------------
    if ctype == "log":
        lines = content.split("\n")

        buffer = []
        size = 0

        for line in lines:
            line_size = len(line)

            # if adding this line exceeds limit → flush buffer
            if size + line_size > max_chars and buffer:
                result.append({
                    "type": "log",
                    "content": "\n".join(buffer)
                })
                buffer = [line]
                size = line_size
            else:
                buffer.append(line)
                size += line_size

        if buffer:
            result.append({
                "type": "log",
                "content": "\n".join(buffer)
            })

    # -------------------------
    # 2. CODE CHUNKING (structure-aware)
    # -------------------------
    elif ctype == "code":
        code_chunks = chunk_code(content)
        result = code_chunks

    # -------------------------
    # 3. fallback (unknown type)
    # -------------------------
    else:
        for i in range(0, len(content), max_chars):
            result.append({
                "type": ctype,
                "content": content[i:i + max_chars]
            })

    return result

def chunk_code(content: str, max_chars: int = 500):
    """
    Final safe code chunker:
    - structure split first
    - strict size enforcement second
    """
    blocks = split_code_into_blocks(content)

    final_chunks = []

    for block in blocks:
        safe_chunks = force_split_block(block, max_chars)
        final_chunks.extend(safe_chunks)

    return [
        {"type": "code", "content": c}
        for c in final_chunks
    ]

def split_code_into_blocks(code: str):
    """
    First pass: split code into logical blocks.
    Keeps functions/classes together when possible.
    """
    pattern = r'(?=^class |^def |^async def |^function )'
    blocks = re.split(pattern, code, flags=re.MULTILINE)

    # clean empty blocks
    return [b.strip() for b in blocks if b.strip()]

def force_split_block(block: str, max_chars: int):
    """
    Second pass: hard split oversized blocks safely.
    """
    if len(block) <= max_chars:
        return [block]

    lines = block.split("\n")
    chunks = []

    buffer = []
    size = 0

    for line in lines:
        line_size = len(line) + 1  # newline

        if size + line_size > max_chars and buffer:
            chunks.append("\n".join(buffer))
            buffer = [line]
            size = line_size
        else:
            buffer.append(line)
            size += line_size

    if buffer:
        chunks.append("\n".join(buffer))

    return chunks

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

    chunks = code_chunks + log_chunks

    # embed_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

    logging.info(f"embedding chunks ...")
    embedding_start = time.time()
    vectors = embed_chunks(chunks)
    embedding_end = time.time()
    logging.info(f"took {embedding_end - embedding_start} to embed all chunks")

    faiss_indexes = build_faiss_index(vectors)

    results = search("why are some requests failing?", faiss_indexes, chunks)
