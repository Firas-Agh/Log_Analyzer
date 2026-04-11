import os
import re
import time
import logging
import ast
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from chunking.chunk_code import chunk_codebase
from chunking.chunk_logs import chunk_logs

if __name__ == "__main__":
    path = "log files/almatarhotels_run_prod_79.log"
    code_repo_dir="C:/Users/user/Desktop/SEERA/github new repo's/crawla_almatarhotels"

    logging.basicConfig(level=logging.INFO)

    size = os.path.getsize(path)
    size_mb = size/1000000

    log_chunks = chunk_logs(path, size_mb)

    code_chunks = chunk_codebase(code_repo_dir)
    code_end_time = time.time()
    logging.info(f"took {code_end_time - code_start_time} to chunk codebase")





