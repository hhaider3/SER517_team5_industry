from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from .wikimedia_image_service import search_wikimedia_images
from .wikipedia_image_service import search_wikipedia_images
from .openverse_image_service import search_openverse_images
from .image_search_service import search_images as search_local_images

logger = logging.getLogger(__name__)


def _fetch_wikimedia(query: str, n: int) -> List[Dict[str, Any]]:
    results = search_wikimedia_images(query, n)
    for r in results:
        r["_source_label"] = "Wikimedia"
    return results


def _fetch_wikipedia(query: str, n: int) -> List[Dict[str, Any]]:
    results = search_wikipedia_images(query, n)
    for r in results:
        r["_source_label"] = "Wikipedia"
    return results


def _fetch_openverse(query: str, n: int) -> List[Dict[str, Any]]:
    results = search_openverse_images(query, n)
    for r in results:
        r["_source_label"] = r.get("source") or "Openverse"
    return results


def _fetch_local(query: str, n: int) -> List[Dict[str, Any]]:
    results = search_local_images(query, n)
    for r in results:
        r["_source_label"] = "Local"
    return results


def _deduplicate(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_urls: set[str] = set()
    unique: List[Dict[str, Any]] = []

    for img in images:
        key = re.sub(r"/\d+px-", "/XXpx-", img.get("thumb_url") or img.get("url", ""))
        if key not in seen_urls:
            seen_urls.add(key)
            unique.append(img)

    return unique


def search_all_images(query_text: str, n_results: int = 6) -> List[Dict[str, Any]]:
    if not query_text or not query_text.strip():
        return []

    per_source = max(2, n_results // 3)

    all_images: List[Dict[str, Any]] = []

    # Sequential fetching (simple version)
    for fetch_fn in [
        _fetch_wikimedia,
        _fetch_wikipedia,
        _fetch_openverse,
        _fetch_local,
    ]:
        try:
            results = fetch_fn(query_text, per_source)
            all_images.extend(results)
        except Exception as e:
            logger.warning(f"Source failed: {e}")

    if not all_images:
        return []

    return _deduplicate(all_images)[:n_results]