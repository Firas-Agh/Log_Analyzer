import ast
import logging
import os
import time

TYPE_MAP = {
    "FunctionDef": "function",
    "AsyncFunctionDef": "function",
    "ClassDef": "class"
}

def chunk_codebase(root_dir):
    logging.info(f"chunking code base ...")
    code_start_time = time.time()
    all_chunks = []

    py_files = get_python_files(root_dir)

    for file_path in py_files:
        try:
            tree, source = parse_file(file_path)
            chunks = extract_chunks(tree, source, file_path)
            all_chunks.extend(chunks)

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    code_end_time = time.time()
    logging.info(f"took {code_end_time - code_start_time} to chunk codebase")

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
                "type": "code",
                "content": code_chunk,
                "metadata":{
                    "id": chunk_id,
                    "code_type": type(node).__name__,
                    "name": node.name,
                    "file": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "parent": parent
                }
            }

            chunks.append(chunk)

    return chunks
