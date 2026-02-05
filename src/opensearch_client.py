"""
OpenSearch: AWS IAM auth, bulk push with size splitting and parallel workers.
"""
import os
import json
import requests
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor


def _get_opensearch_auth(region: str):
    """Build AWS4Auth for OpenSearch (no session token)."""
    try:
        from requests_aws4auth import AWS4Auth
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if not access_key or not secret_key:
            return None
        return AWS4Auth(access_key, secret_key, region, "es")
    except ImportError:
        return None


def _split_documents_by_size(
    documents: List[Dict],
    index_name: str,
    max_bytes: int,
) -> List[List[Dict]]:
    """Split documents so each bulk body is <= max_bytes."""
    if not documents or max_bytes <= 0:
        return [documents] if documents else []
    sub_batches = []
    current = []
    current_size = 0
    for doc in documents:
        line1 = json.dumps({"index": {"_index": index_name, "_id": doc["_id"]}}) + "\n"
        line2 = json.dumps(doc["_source"]) + "\n"
        doc_size = len(line1.encode("utf-8")) + len(line2.encode("utf-8"))
        if current and current_size + doc_size > max_bytes:
            sub_batches.append(current)
            current = []
            current_size = 0
        current.append(doc)
        current_size += doc_size
    if current:
        sub_batches.append(current)
    return sub_batches


def _send_one_bulk(
    documents: List[Dict],
    url: str,
    index_name: str,
    auth,
    timeout: int,
    max_retries: int,
) -> Tuple[int, int]:
    """Send one bulk request. Returns (success_count, failed_count)."""
    bulk_body = ""
    for doc in documents:
        bulk_body += json.dumps({"index": {"_index": index_name, "_id": doc["_id"]}}) + "\n"
        bulk_body += json.dumps(doc["_source"]) + "\n"
    body_bytes = bulk_body.encode("utf-8")
    for attempt in range(max_retries + 1):
        resp = requests.post(
            f"{url}/_bulk",
            data=body_bytes,
            headers={"Content-Type": "application/x-ndjson"},
            auth=auth,
            timeout=timeout,
        )
        if resp.status_code == 429 and attempt < max_retries:
            import time
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        out = resp.json()
        items = out.get("items", [])
        success = sum(1 for it in items if it.get("index", {}).get("status") in (200, 201))
        return success, len(items) - success
    return 0, len(documents)


def push_batch_to_opensearch(documents: List[Dict], config: Dict) -> Tuple[int, int]:
    """
    Push bulk documents to OpenSearch. Splits by max_bulk_bytes, uses bulk_parallel_workers.
    Returns (success_count, failed_count).
    """
    if not documents:
        return 0, 0
    url = (config.get("opensearch_url") or "").strip()
    if not url:
        return 0, len(documents)
    if not url.startswith("http"):
        url = "https://" + url
    index_name = config.get("opensearch_index", "wiki_kb_nested")
    region = config.get("opensearch_region", "ap-south-1")
    max_bulk_bytes = config.get("max_bulk_bytes", 12 * 1024 * 1024)
    timeout = config.get("bulk_request_timeout", 60)
    max_retries = config.get("bulk_max_retries", 5)
    auth = _get_opensearch_auth(region)
    if not auth:
        return 0, len(documents)
    sub_batches = _split_documents_by_size(documents, index_name, max_bulk_bytes)
    total_success, total_failed = 0, 0
    parallel_workers = min(config.get("bulk_parallel_workers", 4), len(sub_batches), 16)
    try:
        if parallel_workers <= 1 or len(sub_batches) <= 1:
            for sub in sub_batches:
                success, failed = _send_one_bulk(sub, url, index_name, auth, timeout, max_retries)
                total_success += success
                total_failed += failed
        else:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [
                    executor.submit(_send_one_bulk, sub, url, index_name, auth, timeout, max_retries)
                    for sub in sub_batches
                ]
                for fut in futures:
                    success, failed = fut.result()
                    total_success += success
                    total_failed += failed
        return total_success, total_failed
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return total_success, len(documents) - total_success


def load_progress(progress_file: str) -> Dict:
    """Load progress for resume."""
    if not progress_file or not os.path.exists(progress_file):
        return {}
    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_progress(
    progress_file: str,
    stats: Dict,
    batch_num: int,
    stream_state: Optional[Dict] = None,
) -> None:
    """Save progress after each batch. stream_state can include current_file, pages_into_current_file for fast resume."""
    if not progress_file:
        return
    os.makedirs(os.path.dirname(progress_file) or ".", exist_ok=True)
    base_batches = stats.get("initial_batches_completed", stats.get("batches_skipped", 0))
    data = {
        "batches_completed": base_batches + stats.get("batches_processed", 0),
        "pages_processed": stats.get("total_pages", 0),
        "documents_pushed": stats.get("total_documents", 0),
        "last_batch": batch_num + 1,
    }
    if stream_state:
        if stream_state.get("current_file"):
            data["last_stream_file"] = stream_state["current_file"]
        if stream_state.get("pages_into_current_file") is not None:
            data["pages_into_current_file"] = stream_state["pages_into_current_file"]
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass
