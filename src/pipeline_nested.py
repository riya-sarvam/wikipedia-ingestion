"""
Nested pipeline: one OpenSearch doc per article, chunks[] (nested) with content + vector.
Stream -> chunk (section/paragraph) -> embed -> push.
"""
import os
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from .dump_reader import stream_pages, batch_pages
from .chunking import parse_sections, clean_text, split_into_chunks
from .embedding import generate_embeddings
from .opensearch_client import push_batch_to_opensearch, load_progress, save_progress


def process_single_page_nested(page: Dict, config: Dict) -> Optional[Dict]:
    """
    Process one page into a single nested doc: article-level fields + chunks[] (no vectors yet).
    """
    page_id = str(page.get("_id", ""))
    title = page.get("title", "")
    timestamp = page.get("timestamp", "")
    skip_sections = [s.lower() for s in config.get("skip_sections", [])]
    include_auxiliary = config.get("include_auxiliary", True)
    chunk_size = config.get("chunk_size", 500)
    chunk_overlap = config.get("chunk_overlap", 50)
    min_chunk_length = config.get("min_chunk_length", 20)

    full_text = page.get("text", "")
    article_hash = hashlib.md5(full_text.encode()).hexdigest()
    chunks_out = []

    for section in parse_sections(page, include_auxiliary=include_auxiliary):
        section_path = section["section_path"]
        section_name = section_path.split(" > ")[-1].lower()
        if any(skip in section_name for skip in skip_sections):
            continue
        clean_content = clean_text(section["content"])
        if not clean_content or len(clean_content) < min_chunk_length:
            continue
        text_chunks = split_into_chunks(clean_content, chunk_size, chunk_overlap)
        for idx, chunk_text in enumerate(text_chunks):
            chunks_out.append({
                "chunk_index": idx,
                "section_path": section_path,
                "content": chunk_text,
                "char_count": len(chunk_text),
            })

    if not chunks_out:
        return None

    return {
        "article_id": page_id,
        "article_title": title,
        "article_hash": article_hash,
        "timestamp": timestamp,
        "total_chunks": len(chunks_out),
        "chunks": chunks_out,
    }


def process_pages_parallel_nested(pages: List[Dict], config: Dict, num_workers: int = 4) -> List[Dict]:
    """Process pages into nested docs. Returns list of nested docs (chunks without vectors)."""
    fn = partial(process_single_page_nested, config=config)
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for doc in executor.map(fn, pages):
            if doc is not None:
                results.append(doc)
    return results


def _ensure_doc_id_length(doc_id: str, max_bytes: int = 512) -> str:
    encoded = doc_id.encode("utf-8")
    if len(encoded) <= max_bytes:
        return doc_id
    return hashlib.sha256(encoded).hexdigest()


def build_bulk_docs_nested(
    nested_docs: List[Dict],
    index_name: str,
    max_doc_id_bytes: int = 512,
) -> List[Dict]:
    """Build list of { _index, _id, _source } for bulk index. Chunks must already have 'vector' set."""
    documents = []
    for doc in nested_docs:
        doc_id = _ensure_doc_id_length(doc["article_id"], max_doc_id_bytes)
        source = {
            "article_id": doc["article_id"],
            "article_title": doc["article_title"],
            "article_hash": doc["article_hash"],
            "timestamp": doc["timestamp"],
            "total_chunks": doc["total_chunks"],
            "chunks": doc["chunks"],
        }
        documents.append({
            "_index": index_name,
            "_id": doc_id,
            "_source": source,
        })
    return documents


def run_pipeline_nested(
    input_file: str,
    config: Optional[Dict] = None,
    max_pages: Optional[int] = None,
    skip_batches: int = 0,
    verbose: bool = True,
) -> Dict:
    """
    Stream pages -> process to nested docs -> embed chunk contents -> push to OpenSearch.
    """
    config = dict(config or {})
    index_name = config.get("opensearch_index") or os.getenv("OPENSEARCH_INDEX", "wiki_kb_nested")
    config["opensearch_index"] = index_name

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    progress_file = config.get("progress_file") or "output/pipeline_progress_nested.json"
    max_doc_id_bytes = config.get("max_doc_id_bytes", 512)

    if progress_file and os.path.exists(progress_file):
        prog = load_progress(progress_file)
        resume_from = prog.get("batches_completed", 0)
        if resume_from > 0:
            skip_batches = max(skip_batches or 0, resume_from)
            if verbose:
                print(f"Resuming: skip first {skip_batches} batches ({prog.get('pages_processed', 0)} pages, {prog.get('documents_pushed', 0)} docs)")

    os.makedirs(os.path.dirname(progress_file) or ".", exist_ok=True)

    stats = {"total_pages": 0, "total_chunks": 0, "total_documents": 0, "batches_processed": 0, "batches_skipped": 0}
    start_time = time.time()
    page_stream = stream_pages(input_file, max_pages=max_pages)
    batched_pages = batch_pages(page_stream, batch_size)

    for batch_num, pages_batch in enumerate(batched_pages):
        if batch_num < skip_batches:
            stats["total_pages"] += len(pages_batch)
            stats["batches_skipped"] += 1
            if verbose:
                print(f"\rSkipping batch {batch_num + 1}/{skip_batches}...", end="", flush=True)
                if batch_num + 1 == skip_batches:
                    print()
            continue

        batch_start = time.time()
        if verbose:
            print(f"  Batch {batch_num + 1}: chunking {len(pages_batch)} pages...", flush=True)
        nested_docs = process_pages_parallel_nested(pages_batch, config, num_workers)
        if not nested_docs:
            stats["total_pages"] += len(pages_batch)
            continue

        all_contents = [c["content"] for doc in nested_docs for c in doc["chunks"]]
        num_chunks = len(all_contents)
        if verbose:
            print(f"  Batch {batch_num + 1}: embedding {num_chunks} chunks...", flush=True)
        embeddings = generate_embeddings(all_contents, config)
        idx = 0
        for doc in nested_docs:
            for c in doc["chunks"]:
                c["vector"] = embeddings[idx]
                idx += 1

        documents = build_bulk_docs_nested(nested_docs, index_name, max_doc_id_bytes)
        success, failed = push_batch_to_opensearch(documents, config)
        if failed and verbose:
            print(f"\n  Push: {success} ok, {failed} failed")

        stats["total_pages"] += len(pages_batch)
        stats["total_chunks"] += sum(d["total_chunks"] for d in nested_docs)
        stats["total_documents"] += len(documents)
        stats["batches_processed"] += 1
        if progress_file:
            save_progress(progress_file, stats, batch_num)

        if verbose:
            batch_time = time.time() - batch_start
            total_time = time.time() - start_time
            print(f"  Batch {batch_num + 1}: {stats['total_pages']} pages, {stats['total_documents']} docs pushed, {batch_time:.1f}s (total {total_time:.1f}s)")

        delay = config.get("batch_delay_seconds", 0)
        if delay and delay > 0:
            time.sleep(delay)

    if verbose:
        print(f"\nPipeline complete. Pages: {stats['total_pages']}, Docs: {stats['total_documents']}")
    return stats
