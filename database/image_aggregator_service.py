"""
image_aggregator_service.py
Fetches images from multiple sources in parallel and merges the results.

Sources (in priority order):
    1. Wikimedia Commons  – educational diagrams and illustrations
    2. Wikipedia          – curated article images
    3. Openverse          – openly-licensed images (Flickr, NASA, museums, etc.)
    4. ChromaDB (local)   – K-12 image database (fallback)

All sources are queried concurrently using threads so the total latency is
roughly equal to the slowest single source (~1-2 s) rather than their sum.

Public API
----------
    search_all_images(query_text, n_results=6) → str  (HTML gallery block)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def build_multi_source_gallery_html(images: List[Dict[str, Any]]) -> str:
    if not images:
        return ""

    cards = []
    for img in images:
        thumb = img.get("thumb_url") or img.get("url", "")
        full = img.get("article_url") or img.get("url", "")
        title = img.get("title") or img.get("topic") or "Image"
        source = img.get("_source_label") or ""

        # Sanitise for HTML attributes
        title_safe = re.sub(r"[<>&\"']", "", str(title))
        source_safe = re.sub(r"[<>&\"']", "", str(source))

        caption = title_safe
        if source_safe:
            caption += f' <span class="chat-image-source">({source_safe})</span>'

        cards.append(
            f'<a href="{full}" target="_blank" rel="noopener" '
            f'class="chat-image-card" title="{title_safe}">'
            f'<img src="{thumb}" alt="{title_safe}" loading="lazy">'
            f'<span class="chat-image-caption">{caption}</span>'
            f"</a>"
        )

    return (
        '<div class="chat-image-gallery">'
        '<div class="chat-image-gallery-header">📷 Related Images</div>'
        + "".join(cards)
        + "</div>"
    )

def search_all_images(query_text: str, n_results: int = 6) -> List[Dict[str, Any]]:

    if not query_text or not query_text.strip():
        return ""

    per_source = max(2, n_results // 3)

    fetchers = {
        "wikimedia": (_fetch_wikimedia, query_text, per_source),
        "wikipedia": (_fetch_wikipedia, query_text, per_source),
        "openverse": (_fetch_openverse, query_text, per_source),
        "local":     (_fetch_local,     query_text, per_source),
    }

    all_images: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fn, q, n): name
            for name, (fn, q, n) in fetchers.items()
        }
        for future in as_completed(futures, timeout=12):
            source_name = futures[future]
            try:
                results = future.result()
                logger.info("Image source '%s' returned %d results", source_name, len(results))
                all_images.extend(results)
            except Exception as exc:
                logger.warning("Image source '%s' failed: %s", source_name, exc)

    if not all_images:
        return ""

    # Deduplicate and cap
    unique = _deduplicate(all_images)[:n_results]

    logger.info(
        "Image aggregator: %d unique images for '%s' (from %d total)",
        len(unique), query_text[:60], len(all_images),
    )

    return build_multi_source_gallery_html(unique)