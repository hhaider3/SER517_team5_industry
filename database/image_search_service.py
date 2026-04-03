"""
image_search_service.py
Bridge between the Django chatbot and the K-12 ChromaDB image search.

Provides a single public function:
    search_images(query_text, n_results=3, subject=None, grade=None)
        → list[dict]   [{url, thumb_url, subject, topic, score}, …]

The ChromaDB collection is lazily initialised once per process.
All errors are caught so the chat flow is never broken.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.conf import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton for the ChromaDB collection
# ---------------------------------------------------------------------------
_collection = None
_init_failed = False


def _get_collection():
    """Return the ChromaDB collection, initialising on first call."""
    global _collection, _init_failed

    if _collection is not None:
        return _collection
    if _init_failed:
        return None

    try:
        import chromadb
        from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

        try:
            from chromadb.utils.data_loaders import ImageLoader
        except Exception:
            ImageLoader = None

        db_path = getattr(settings, "IMAGE_DB_PATH", None)
        if not db_path:
            logger.warning("IMAGE_DB_PATH is not set in Django settings – image search disabled.")
            _init_failed = True
            return None

        if not Path(db_path).exists():
            logger.warning("IMAGE_DB_PATH %s does not exist – image search disabled.", db_path)
            _init_failed = True
            return None

        client = chromadb.PersistentClient(path=str(db_path))
        embedder = OpenCLIPEmbeddingFunction()

        kwargs: Dict[str, Any] = dict(
            name="k12_education_images",
            embedding_function=embedder,
        )
        if ImageLoader:
            kwargs["data_loader"] = ImageLoader()

        _collection = client.get_or_create_collection(**kwargs)
        logger.info(
            "K-12 image collection ready – %d images in %s",
            _collection.count(),
            db_path,
        )
        return _collection

    except Exception as exc:
        logger.error("Failed to initialise K-12 image search: %s", exc)
        _init_failed = True
        return None


# ---------------------------------------------------------------------------
# Scoring helpers (mirrored from search_k12_db.py)
# ---------------------------------------------------------------------------
def _to_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return default


def _grade_penalty(grade: Optional[int], gmin: Optional[int], gmax: Optional[int]) -> float:
    if grade is None:
        return 0.0
    if gmin is None or gmax is None:
        return 0.15
    if gmin <= grade <= gmax:
        return 0.0
    d = min(abs(grade - gmin), abs(grade - gmax))
    return min(1.5, 0.20 * d)


def _combine_score(distance: float, penalty: float) -> float:
    sim = 1.0 / (1.0 + max(0.0, float(distance)))
    return sim - penalty


# ---------------------------------------------------------------------------
# URI → Django-servable URL
# ---------------------------------------------------------------------------
def _uri_to_url(uri: str, thumb: bool = False) -> Optional[str]:
    """Convert an on-disk image path into a ``/media/…`` URL.

    ChromaDB stores URIs like ``/abs/path/to/k12_images/full/abc123.jpg``.
    We need to make this relative to ``MEDIA_ROOT`` so Django can serve it.
    If *thumb* is True we swap ``/full/`` → ``/thumb/`` in the path.
    """
    media_root = getattr(settings, "MEDIA_ROOT", None)
    if not media_root:
        return None

    p = Path(uri)
    if not p.exists():
        # Try resolving relative to MEDIA_ROOT
        candidate = Path(media_root) / Path(uri).name
        if candidate.exists():
            p = candidate
        else:
            return None

    if thumb:
        # Swap full → thumb in the path
        parts = list(p.parts)
        try:
            idx = parts.index("full")
            parts[idx] = "thumb"
            thumb_p = Path(*parts)
            if thumb_p.exists():
                p = thumb_p
        except ValueError:
            pass  # no "full" directory component – use original

    try:
        rel = p.relative_to(Path(media_root))
    except ValueError:
        # URI is not inside MEDIA_ROOT – serve via absolute-ish fallback
        return None

    media_url = getattr(settings, "MEDIA_URL", "/media/")
    return f"{media_url}{rel}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def search_images(
    query_text: str,
    n_results: int = 3,
    subject: Optional[str] = None,
    grade: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Search the K-12 image database for images related to *query_text*.

    Returns a list of dicts (may be empty):
        [{"url": str, "thumb_url": str, "subject": str, "topic": str, "score": float}, …]
    """
    if not query_text or not query_text.strip():
        return []

    col = _get_collection()
    if col is None or col.count() == 0:
        return []

    try:
        # Build optional metadata filter
        where_clauses: List[Dict[str, Any]] = []
        if subject:
            where_clauses.append({"subject": subject})

        where: Optional[Dict[str, Any]] = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        raw_k = max(n_results * 5, n_results)

        query_kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": raw_k,
            "include": ["metadatas", "distances", "uris"],
        }
        if where:
            query_kwargs["where"] = where

        res = col.query(**query_kwargs)

        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        mds = (res.get("metadatas") or [[]])[0]
        uris = (res.get("uris") or [[]])[0]

        if not ids:
            return []

        # Rank with hybrid keyword boost (same logic as search_k12_db.py)
        ranked = []
        query_words = set(query_text.lower().split())

        for dist, uri, md in zip(dists, uris, mds):
            gmin = _to_int(md.get("grade_min"), None)
            gmax = _to_int(md.get("grade_max"), None)
            pen = _grade_penalty(grade, gmin, gmax)
            score = _combine_score(float(dist), pen)

            # Keyword boost
            topic = str(md.get("topic") or "").lower()
            subj = str(md.get("subject") or "").lower()
            desc = str(md.get("description") or "").lower()
            for w in query_words:
                if len(w) > 3:
                    if w in topic:
                        score += 0.15
                    if w in subj:
                        score += 0.10
                    if w in desc:
                        score += 0.05

            ranked.append((score, uri, md))

        ranked.sort(key=lambda x: x[0], reverse=True)
        ranked = ranked[:n_results]

        # Convert to output dicts
        results = []
        for score, uri, md in ranked:
            full_url = _uri_to_url(uri, thumb=False)
            thumb_url = _uri_to_url(uri, thumb=True) or full_url
            if not full_url:
                continue
            results.append({
                "url": full_url,
                "thumb_url": thumb_url,
                "subject": md.get("subject", ""),
                "topic": md.get("topic", ""),
                "score": round(score, 3),
            })

        logger.info("Image search for '%s' returned %d results", query_text[:60], len(results))
        return results

    except Exception as exc:
        logger.error("Image search failed for '%s': %s", query_text[:60], exc)
        return []


def build_image_gallery_html(images: List[Dict[str, Any]]) -> str:
    """Render a list of image result dicts into an HTML gallery block.

    Returns an empty string if *images* is empty.
    """
    if not images:
        return ""

    cards = []
    for img in images:
        thumb = img.get("thumb_url") or img.get("url", "")
        full = img.get("url", "")
        topic = img.get("topic") or img.get("subject") or "Image"
        # Sanitise for HTML attributes
        topic_safe = re.sub(r"[<>&\"']", "", topic)
        cards.append(
            f'<a href="{full}" target="_blank" rel="noopener" class="chat-image-card" title="{topic_safe}">'
            f'<img src="{thumb}" alt="{topic_safe}" loading="lazy">'
            f'<span class="chat-image-caption">{topic_safe}</span>'
            f"</a>"
        )

    return (
        '<div class="chat-image-gallery">'
        '<div class="chat-image-gallery-header">📷 Related Images</div>'
        + "".join(cards)
        + "</div>"
    )