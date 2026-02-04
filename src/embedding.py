"""
Azure (and optional custom) embedding API with parallel workers and token truncation.
"""
import time
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor


MAX_CHARS = 10000  # Stay under 8192 token limit (~1.2 chars/token)


def _send_one_embedding_batch_azure(
    batch: List[str],
    url: str,
    headers: Dict,
    timeout: int,
    max_retries: int,
) -> List[List[float]]:
    """Send one batch to Azure embedding API. Returns embeddings for this batch."""
    if not batch:
        return []
    for attempt in range(max_retries + 1):
        response = requests.post(
            url,
            headers=headers,
            json={"input": batch},
            timeout=timeout,
        )
        if response.status_code == 429 and attempt < max_retries:
            time.sleep(2 ** attempt)
            continue
        break
    if response.status_code != 200:
        raise RuntimeError(f"Azure API error {response.status_code}: {response.text[:500]}")
    data = response.json()
    if "data" in data:
        sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in sorted_data]
    if "embeddings" in data:
        return data["embeddings"]
    raise RuntimeError(f"Unexpected response format: {list(data.keys())}")


def generate_embeddings_azure(texts: List[str], config: Dict) -> List[List[float]]:
    """
    Generate embeddings via Azure. Truncates text over 10000 chars to stay under 8192 tokens.
    When embedding_parallel_workers > 1, sends sub-batches concurrently.
    """
    safe_texts = []
    for t in texts:
        if len(t) > MAX_CHARS:
            truncated = t[:MAX_CHARS]
            last_period = truncated.rfind(". ")
            if last_period > MAX_CHARS * 0.7:
                truncated = truncated[: last_period + 1]
            safe_texts.append(truncated)
        else:
            safe_texts.append(t)
    texts = safe_texts

    endpoint = config.get("azure_endpoint")
    api_key = config.get("azure_api_key")
    api_version = config.get("azure_api_version", "2024-02-01")
    batch_size = config.get("embedding_batch_size", 32)
    timeout = config.get("embedding_timeout", 120)
    max_retries = config.get("embedding_max_retries", 5)
    parallel_workers = config.get("embedding_parallel_workers", 1)

    if not endpoint or not api_key:
        raise ValueError("Azure endpoint/key not configured. Set in .env.")

    is_openai = "openai.azure.com" in endpoint.lower()
    url = f"{endpoint}?api-version={api_version}" if is_openai and "?" not in endpoint else endpoint
    headers = {"Content-Type": "application/json", "api-key": api_key} if is_openai else {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    sub_batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    all_embeddings = []

    if parallel_workers <= 1 or len(sub_batches) <= 1:
        for batch in sub_batches:
            all_embeddings.extend(
                _send_one_embedding_batch_azure(batch, url, headers, timeout, max_retries)
            )
    else:
        workers = min(parallel_workers, len(sub_batches), 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _send_one_embedding_batch_azure,
                    batch,
                    url,
                    headers,
                    timeout,
                    max_retries,
                )
                for batch in sub_batches
            ]
            for fut in futures:
                all_embeddings.extend(fut.result())

    return all_embeddings


def truncate_text(text: str, max_tokens: int = 6000) -> str:
    """Truncate to fit token limit. ~2.5 chars per token."""
    max_chars = int(max_tokens * 2.5)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.7:
        return truncated[: last_period + 1]
    return truncated


def generate_embeddings(texts: List[str], config: Dict) -> List[List[float]]:
    """Generate embeddings using configured provider (Azure). Truncates long texts."""
    max_tokens = config.get("max_chunk_tokens", 6000)
    truncated_texts = [truncate_text(t, max_tokens) for t in texts]
    provider = config.get("embedding_provider", "azure")
    if provider == "azure":
        return generate_embeddings_azure(truncated_texts, config)
    # Custom endpoint: optional
    endpoint = config.get("embedding_endpoint", "")
    if endpoint:
        import requests as req
        batch_size = config.get("embedding_batch_size", 32)
        timeout = config.get("embedding_timeout", 120)
        all_embeddings = []
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i : i + batch_size]
            r = req.post(endpoint, json={"texts": batch}, timeout=timeout)
            r.raise_for_status()
            all_embeddings.extend(r.json().get("embeddings", []))
        return all_embeddings
    raise ValueError("No embedding provider configured. Set Azure vars or embedding_endpoint in .env.")
