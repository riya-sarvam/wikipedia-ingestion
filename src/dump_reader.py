"""
Stream pages from Wikipedia CirrusSearch dumps (.json.gz or .json.bz2).
Single file or directory of files; yields one page at a time.
Supports resume: start from a specific file and skip N pages in that file.
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
    skip_first_n: int = 0,
) -> Generator[Dict, None, None]:
    """Yield pages from a single CirrusSearch dump file. Optionally skip first N valid pages."""
    if filepath.endswith(".gz"):
        open_func = gzip.open
    elif filepath.endswith(".bz2"):
        open_func = bz2.open
    else:
        open_func = open

    skipped = [0]  # mutable: count of pages skipped so far

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
                if skipped[0] < skip_first_n:
                    skipped[0] += 1
                    count_so_far[0] += 1
                    continue
                yield content
                count_so_far[0] += 1
                if max_pages is not None and count_so_far[0] >= max_pages:
                    return
            except json.JSONDecodeError:
                continue


def _list_stream_files(folder_path: str, extensions: tuple = (".json.bz2", ".json.gz")) -> List[str]:
    """Return sorted list of stream file paths in folder."""
    folder = os.path.abspath(folder_path)
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and any(name.endswith(ext) for ext in extensions):
            files.append(path)
    return files


def stream_pages(
    filepath: str,
    max_pages: Optional[int] = None,
    resume_file: Optional[str] = None,
    resume_skip_pages: int = 0,
) -> Generator[Dict, None, None]:
    """
    Stream pages from CirrusSearch dump (file or directory of .json.gz/.json.bz2).
    Yields one page at a time. Main namespace only; skips redirects.
    If resume_file and resume_skip_pages are set, start from that file and skip that many pages (fast resume).
    """
    if os.path.isdir(filepath):
        yield from stream_pages_from_folder(
            filepath,
            max_pages=max_pages,
            start_from_file_path=resume_file,
            skip_pages_in_first_file=resume_skip_pages,
        )
        return
    count_so_far = [0]
    yield from _stream_pages_from_file(filepath, max_pages, count_so_far, skip_first_n=resume_skip_pages)


def stream_pages_from_folder(
    folder_path: str,
    max_pages: Optional[int] = None,
    extensions: tuple = (".json.bz2", ".json.gz"),
    start_from_file_path: Optional[str] = None,
    skip_pages_in_first_file: int = 0,
) -> Generator[Dict, None, None]:
    """Stream from CirrusSearch dumps in a folder (sorted order). Can start from a specific file and skip N pages in it."""
    files = _list_stream_files(folder_path, extensions)
    if not files:
        return

    start_idx = 0
    skip_in_first = 0
    if start_from_file_path:
        resume_basename = os.path.basename(start_from_file_path)
        for i, path in enumerate(files):
            if path == start_from_file_path or os.path.basename(path) == resume_basename:
                start_idx = i
                skip_in_first = skip_pages_in_first_file
                break

    count_so_far = [0]
    for idx, path in enumerate(files[start_idx:], start=start_idx):
        if max_pages is not None and count_so_far[0] >= max_pages:
            return
        use_skip = skip_in_first if idx == start_idx else 0
        yield from _stream_pages_from_file(path, max_pages, count_so_far, skip_first_n=use_skip)


def stream_pages_tracked(
    filepath: str,
    max_pages: Optional[int],
    state: Dict,
    resume_file: Optional[str] = None,
    resume_skip_pages: int = 0,
) -> Generator[Dict, None, None]:
    """
    Like stream_pages but updates state as we go: state['current_file'], state['pages_into_current_file'].
    Caller can read state after each batch to save progress. Resume uses resume_file + resume_skip_pages.
    """
    state["current_file"] = None
    state["pages_into_current_file"] = 0

    if os.path.isdir(filepath):
        files = _list_stream_files(filepath)
        if not files:
            return
        start_idx = 0
        skip_in_first = 0
        if resume_file:
            resume_basename = os.path.basename(resume_file)
            for i, p in enumerate(files):
                if p == resume_file or os.path.basename(p) == resume_basename:
                    start_idx = i
                    skip_in_first = resume_skip_pages
                    break
        count_so_far = [0]
        for idx, path in enumerate(files[start_idx:], start=start_idx):
            if max_pages is not None and count_so_far[0] >= max_pages:
                return
            state["current_file"] = path
            use_skip = skip_in_first if idx == start_idx else 0
            pages_in_this_file = [0]
            for page in _stream_pages_from_file(path, max_pages, count_so_far, skip_first_n=use_skip):
                pages_in_this_file[0] += 1
                state["pages_into_current_file"] = pages_in_this_file[0]
                yield page
        return

    # Single file
    state["current_file"] = filepath
    count_so_far = [0]
    pages_in_file = [0]
    for page in _stream_pages_from_file(filepath, max_pages, count_so_far, skip_first_n=resume_skip_pages):
        pages_in_file[0] += 1
        state["pages_into_current_file"] = pages_in_file[0]
        yield page


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
