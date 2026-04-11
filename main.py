import os
import re
import time
import logging
import ast

#Chunking log file
LOG_START_PATTERN = re.compile(r"^\[\d{4}-\d{2}-\d{2}T.*Z\]\s\d{4}-\d{2}-\d{2}\s.*\s\[.*\]")

def load_log_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()

def chunk_logs_semantic(lines):
    chunks = []
    current_chunk = []
    in_log_section = False

    for line in lines:
        line = line.rstrip("\n")

        # Check if this line starts a new log message
        if LOG_START_PATTERN.match(line):
            in_log_section = True
            # Save previous chunk if exists
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []

            current_chunk.append(line)
        else:
            if in_log_section:
                # continuation of log message
                current_chunk.append(line)
            else:
                # pre-log normal text → chunk by paragraphs
                if line.strip() == "":
                    if current_chunk:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
                else:
                    current_chunk.append(line)

    # Add last chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def extract_metadata(chunk):
    first_line = chunk.split("\n")[0]

    timestamp_match = re.search(r"\[(.*?)\]", first_line)
    level_match = re.search(r"\b(INFO|DEBUG|ERROR|WARNING)\b", first_line)

    return {
        "timestamp": timestamp_match.group(1) if timestamp_match else None,
        "level": level_match.group(1) if level_match else None,
        "source": first_line
    }

def get_metadata(chunks):
    metadata = []
    for i, chunk in enumerate(chunks):
        meta = extract_metadata(chunk)
        meta["text"] = chunk
        meta["chunk_id"] = i
        metadata.append(meta)

    return metadata

#Chunking code-base
TYPE_MAP = {
    "FunctionDef": "function",
    "AsyncFunctionDef": "function",
    "ClassDef": "class"
}

def chunk_codebase(root_dir):
    all_chunks = []

    py_files = get_python_files(root_dir)

    for file_path in py_files:
        try:
            tree, source = parse_file(file_path)
            chunks = extract_chunks(tree, source, file_path)
            all_chunks.extend(chunks)

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    return all_chunks

def get_python_files(root_dir):
    py_files = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))

    return py_files


def parse_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    return tree, source


def extract_chunks(tree, source, file_path):
    chunks = []
    parent_map = {}

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    def get_parent_context(node):
        parent = parent_map.get(node)

        while parent:
            if isinstance(parent, ast.ClassDef):
                return f"class:{parent.name}"
            if isinstance(parent, ast.FunctionDef):
                return f"function:{parent.name}"
            parent = parent_map.get(parent)

        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = node.end_lineno  # Python 3.8+

            code_chunk = "\n".join(
                source.splitlines()[start_line - 1:end_line]
            )

            parent = get_parent_context(node)
            chunk_id = f"{file_path}:{node.name}:{start_line}"

            chunk = {
                "id": chunk_id,
                "type": type(node).__name__,
                "name": node.name,
                "file": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "code": code_chunk,
                "parent": parent
            }

            chunks.append(chunk)

    return chunks

if __name__ == "__main__":
    path = "log files/almatarhotels_run_prod_79.log"
    code_repo_dir="C:/Users/user/Desktop/SEERA/github new repo's/crawla_almatarhotels"

    logging.basicConfig(level=logging.INFO)

    size = os.path.getsize(path)
    size_mb = size/1000000

    logging.info(f"extracting lines ...")
    lines = load_log_file(path)

    logging.info(f"chunking logs ...")
    chunking_start_time=time.time()
    log_chunks = chunk_logs_semantic(lines)
    chunking_end_time = time.time()

    logging.info(f"took {chunking_end_time - chunking_start_time} to chunk {size_mb} MB file")

    logging.info(f"extracting metadata ...")
    metadata_start_time = time.time()
    metadata = get_metadata(log_chunks)
    metadata_end_time = time.time()

    logging.info(f"took {metadata_end_time - metadata_start_time} to extract metadata: {len(metadata)}")

    logging.info(f"chunking code base ...")
    code_start_time = time.time()
    code_chunks = chunk_codebase(code_repo_dir)
    code_end_time = time.time()
    logging.info(f"took {code_end_time - code_start_time} to chunk codebase")





