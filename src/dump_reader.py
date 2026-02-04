"""
Stream pages from Wikipedia CirrusSearch dumps (.json.gz or .json.bz2).
Single file or directory of files; yields one page at a time.
"""
import os
import json
import gzip
import bz2
from typing import Dict, Generator, List, Optional


def _stream_pages_from_file(
    filepath: str,
    max_pages: Optional[int],
    count_so_far: list,
) -> Generator[Dict, None, None]:
    """Yield pages from a single CirrusSearch dump file."""
    if filepath.endswith(".gz"):
        open_func = gzip.open
    elif filepath.endswith(".bz2"):
        open_func = bz2.open
    else:
        open_func = open

    with open_func(filepath, "rt", encoding="utf-8") as f:
        while True:
            if max_pages is not None and count_so_far[0] >= max_pages:
                return
            meta_line = f.readline()
            if not meta_line:
                break
            content_line = f.readline()
            if not content_line:
                break
            try:
                meta = json.loads(meta_line)
                content = json.loads(content_line)
                content["_id"] = meta.get("index", {}).get("_id")
                if content.get("namespace", 0) != 0:
                    continue
                if content.get("redirect"):
                    continue
                yield content
                count_so_far[0] += 1
                if max_pages is not None and count_so_far[0] >= max_pages:
                    return
            except json.JSONDecodeError:
                continue


def stream_pages(
    filepath: str,
    max_pages: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Stream pages from CirrusSearch dump (file or directory of .json.gz/.json.bz2).
    Yields one page at a time. Main namespace only; skips redirects.
    """
    if os.path.isdir(filepath):
        yield from stream_pages_from_folder(filepath, max_pages=max_pages)
        return
    count_so_far = [0]
    yield from _stream_pages_from_file(filepath, max_pages, count_so_far)


def stream_pages_from_folder(
    folder_path: str,
    max_pages: Optional[int] = None,
    extensions: tuple = (".json.bz2", ".json.gz"),
) -> Generator[Dict, None, None]:
    """Stream from all CirrusSearch dumps in a folder (sorted order)."""
    folder = os.path.abspath(folder_path)
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and any(name.endswith(ext) for ext in extensions):
            files.append(path)
    count_so_far = [0]
    for path in files:
        if max_pages is not None and count_so_far[0] >= max_pages:
            return
        yield from _stream_pages_from_file(path, max_pages, count_so_far)


def batch_pages(
    page_generator: Generator[Dict, None, None],
    batch_size: int,
) -> Generator[List[Dict], None, None]:
    """Group streamed pages into batches."""
    batch = []
    for page in page_generator:
        batch.append(page)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
