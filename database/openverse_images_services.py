
"""
openverse_image_service.py
Fetches openly-licensed images from Openverse (by WordPress / Creative Commons).

Openverse aggregates images from dozens of providers including Flickr, Europeana,
Smithsonian, NASA, museums, and government archives.  Free, no API key required.

Public API
----------
    search_openverse_images(query, n_results=3) → list[dict]
    build_openverse_gallery_html(images) → str
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://api.openverse.org/v1/images/"
_TIMEOUT = 8  # seconds


def search_openverse_images(
    query_text: str,
    n_results: int = 3,
) -> List[Dict[str, Any]]:
    """Search Openverse for openly-licensed images matching *query_text*.

    Returns a list of dicts:
        [{"url": str, "thumb_url": str, "title": str,
          "creator": str, "source": str, "license": str}, …]
    """
    if not query_text or not query_text.strip():
        return []

    try:
        # Clean up the question to make a better image search query
        clean = re.sub(
            r"\b(what|how|why|when|where|who|is|are|do|does|can|could|would|"
            r"should|the|a|an|of|in|to|for|and|or|it|this|that|please|explain|"
            r"tell|me|about|show|look|like|some|any|much|many)\b",
            "",
            query_text.lower(),
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"\s+", " ", clean).strip()
        search_query = clean or query_text

        params = {
            "q": search_query,
            "page_size": n_results,
            "mature": "false",
        }

        resp = requests.get(
            _API_URL,
            params=params,
            headers={"User-Agent": "AdaptiveTutorBot/1.0 (educational chatbot)"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        results: List[Dict[str, Any]] = []
        for item in data.get("results", []):
            full_url = item.get("url", "")
            thumb_url = item.get("thumbnail") or full_url
            title = item.get("title") or "Image"

            # Skip items without a usable URL
            if not full_url:
                continue

            results.append({
                "url": full_url,
                "thumb_url": thumb_url,
                "title": title[:80],
                "creator": item.get("creator") or "Unknown",
                "source": item.get("source") or item.get("provider") or "Openverse",
                "license": item.get("license") or "",
            })

            if len(results) >= n_results:
                break

        logger.info(
            "Openverse search for '%s' returned %d results",
            query_text[:60],
            len(results),
        )
        return results

    except Exception as exc:
        logger.error("Openverse image search failed for '%s': %s", query_text[:60], exc)
        return []


def build_openverse_gallery_html(images: List[Dict[str, Any]]) -> str:
    """Render Openverse image results into an HTML gallery block.

    Returns empty string if *images* is empty.
    """
    if not images:
        return ""

    cards = []
    for img in images:
        thumb = img.get("thumb_url") or img.get("url", "")
        full = img.get("url", "")
        title = img.get("title") or "Image"
        source = img.get("source") or ""
        # Sanitise for HTML attributes
        title_safe = re.sub(r"[<>&\"']", "", title)
        source_safe = re.sub(r"[<>&\"']", "", source)
        caption = f"{title_safe}"
        if source_safe:
            caption += f" ({source_safe})"

        cards.append(
            f'<a href="{full}" target="_blank" rel="noopener" class="chat-image-card" title="{title_safe}">'
            f'<img src="{thumb}" alt="{title_safe}" loading="lazy">'
            f'<span class="chat-image-caption">{caption}</span>'
            f"</a>"
        )

    return (
        '<div class="chat-image-gallery">'
        '<div class="chat-image-gallery-header">📷 Related Images (Openverse)</div>'
        + "".join(cards)
        + "</div>"
    )