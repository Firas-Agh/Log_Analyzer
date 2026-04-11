import logging
import re
import time

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
        meta["type"] = "log"
        metadata.append(meta)

    return metadata

def chunk_logs(path, size):
    logging.info(f"extracting lines ...")
    lines = load_log_file(path)

    logging.info(f"chunking logs ...")
    chunking_start_time = time.time()
    log_chunks = chunk_logs_semantic(lines)
    chunking_end_time = time.time()

    logging.info(f"took {chunking_end_time - chunking_start_time} to chunk {size} MB file")

    logging.info(f"extracting logs metadata ...")
    metadata_start_time = time.time()
    metadata = get_metadata(log_chunks)
    metadata_end_time = time.time()

    logging.info(f"took {metadata_end_time - metadata_start_time} to extract logs metadata: {len(metadata)}")

    return metadata