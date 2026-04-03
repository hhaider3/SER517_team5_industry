"""
wikimedia_image_service.py
Fetches relevant images from Wikimedia Commons for a given query.

Uses the free MediaWiki API — no API key needed.

Public API
----------
    search_wikimedia_images(query, n_results=3) → list[dict]
    build_wikimedia_gallery_html(images) → str
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://commons.wikimedia.org/w/api.php"
_TIMEOUT = 8  # seconds
_THUMB_WIDTH = 300


def search_wikimedia_images(
    query_text: str,
    n_results: int = 3,
) -> List[Dict[str, Any]]:
    """Search Wikimedia Commons for images matching *query_text*.

    Returns a list of dicts:
        [{"url": str, "thumb_url": str, "title": str, "description": str}, …]
    """
    if not query_text or not query_text.strip():
        return []

    try:
        # Clean up the question to make a better image search query
        clean = re.sub(r"\b(what|how|why|when|where|who|is|are|do|does|can|could|would|should|the|a|an|of|in|to|for|and|or|it|this|that|please|explain|tell|me|about|show)\b", "", query_text.lower(), flags=re.IGNORECASE)
        clean = re.sub(r"\s+", " ", clean).strip()
        search_query = f"{clean} diagram" if clean else query_text

        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": search_query,
            "gsrnamespace": 6,  # File namespace
            "gsrlimit": n_results * 3,  # Fetch extras to filter
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime",
            "iiurlwidth": _THUMB_WIDTH,
            "format": "json",
            "formatversion": 2,
        }

        resp = requests.get(
            _API_URL,
            params=params,
            headers={"User-Agent": "AdaptiveTutorBot/1.0 (educational chatbot)"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        pages = (data.get("query") or {}).get("pages") or []
        results: List[Dict[str, Any]] = []

        for page in pages:
            info_list = page.get("imageinfo", [])
            if not info_list:
                continue

            info = info_list[0]
            mime = info.get("mime", "")

            # Only keep actual images
            if not mime.startswith("image/"):
                continue

            # Skip SVGs that render poorly as thumbnails in some cases
            full_url = info.get("url", "")
            thumb_url = info.get("thumburl") or full_url

            # Extract description from extmetadata
            ext = info.get("extmetadata") or {}
            desc_obj = ext.get("ImageDescription") or {}
            description = _strip_html(desc_obj.get("value", ""))

            title = page.get("title", "").replace("File:", "").rsplit(".", 1)[0]
            # Clean up title
            title = title.replace("_", " ")

            if full_url:
                results.append({
                    "url": full_url,
                    "thumb_url": thumb_url,
                    "title": title[:80],
                    "description": description[:150] if description else "",
                })

            if len(results) >= n_results:
                break

        logger.info("Wikimedia search for '%s' returned %d results", query_text[:60], len(results))
        return results

    except Exception as exc:
        logger.error("Wikimedia image search failed for '%s': %s", query_text[:60], exc)
        return []


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def build_wikimedia_gallery_html(images: List[Dict[str, Any]]) -> str:
    """Render Wikimedia image results into an HTML gallery block.

    Returns empty string if *images* is empty.
    """
    if not images:
        return ""

    cards = []
    for img in images:
        thumb = img.get("thumb_url") or img.get("url", "")
        full = img.get("url", "")
        title = img.get("title") or "Image"
        # Sanitise for HTML attributes
        title_safe = re.sub(r"[<>&\"']", "", title)

        cards.append(
            f'<a href="{full}" target="_blank" rel="noopener" class="chat-image-card" title="{title_safe}">'
            f'<img src="{thumb}" alt="{title_safe}" loading="lazy">'
            f'<span class="chat-image-caption">{title_safe}</span>'
            f"</a>"
        )

    return (
        '<div class="chat-image-gallery">'
        '<div class="chat-image-gallery-header">📷 Related Images</div>'
        + "".join(cards)
        + "</div>"
    )