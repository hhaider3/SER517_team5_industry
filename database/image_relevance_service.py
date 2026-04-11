
"""
image_relevance_service.py
Determines whether a user question would benefit from accompanying images.

Uses a fast LLM call to classify the question, with a keyword-based fallback
in case the LLM is unavailable. This prevents the chatbot from showing
irrelevant images for greetings, abstract questions, or meta-conversations.

Public API
----------
    should_show_images(question: str) → bool
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ava_apps.core.services.llm_client import get_client, get_fast_model

logger = logging.getLogger(__name__)

# ── Fast keyword heuristics (used as fallback if LLM is unavailable) ──────

# Questions that almost never benefit from images
_SKIP_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*(hi|hello|hey|howdy|good\s+(morning|afternoon|evening|night)|greetings)\b",
        r"^\s*(thanks|thank\s*you|thx|ty)\b",
        r"^\s*(bye|goodbye|see\s+you|later)\b",
        r"^\s*(yes|no|ok|okay|sure|got\s+it|i\s+see|understood)\s*[.!?]*\s*$",
        r"^\s*(who|what)\s+(are|r)\s+you\b",
        r"\b(how\s+are\s+you|how\s+do\s+you\s+feel)\b",
        r"\b(repeat|say\s+that\s+again|come\s+again)\b",
    ]
]

# Topics that strongly benefit from images
_VISUAL_KEYWORDS: set[str] = {
    "diagram", "graph", "chart", "picture", "image", "photo",
    "illustration", "shape", "geometry", "triangle", "circle",
    "square", "rectangle", "polygon", "angle", "line",
    "map", "globe", "planet", "solar system", "cell",
    "anatomy", "skeleton", "organ", "plant", "animal",
    "molecule", "atom", "circuit", "wave", "spectrum",
    "volcano", "earthquake", "rock", "mineral", "fossil",
    "flag", "symbol", "color", "colour", "painting",
    "microscope", "telescope", "lens", "mirror", "shadow",
    "fraction", "number line", "bar graph", "pie chart",
    "histogram", "coordinate", "axis", "slope",
    "food chain", "water cycle", "life cycle", "ecosystem",
    "continent", "ocean", "mountain", "river", "weather",
    "butterfly", "insect", "bird", "fish", "mammal",
    "photosynthesis", "mitosis", "dna", "periodic table",
}


def _keyword_heuristic(question: str) -> bool:
    """Return True if keyword signals suggest images would help."""
    q = question.lower()

    # Definite skip
    for pat in _SKIP_PATTERNS:
        if pat.search(q):
            return False

    # Definite show
    for kw in _VISUAL_KEYWORDS:
        if kw in q:
            return True

    # For anything else, default to showing images (the LLM gate is
    # preferred; this only fires when the LLM is unavailable).
    # Short inputs (< 4 words) are likely not visual.
    return len(q.split()) >= 4


# ── LLM-based classification ─────────────────────────────────────────────

_CLASSIFY_PROMPT = """\
You are a classifier that decides whether an educational question would \
benefit from accompanying images, diagrams, or visual aids.

Respond with EXACTLY one word: YES or NO.

Rules:
- YES for questions about visual, spatial, scientific, or concrete topics \
(e.g. shapes, anatomy, maps, graphs, experiments, animals, planets).
- NO for greetings, feelings, opinions, abstract philosophy, meta-questions \
about the chatbot, or very short affirmations like "ok", "thanks", "yes".
- When in doubt, lean YES for educational content, NO for social chit-chat.

Question: {question}
"""


def _llm_classify(question: str) -> Optional[bool]:
    """Ask the fast LLM model whether images are warranted.

    Returns True/False, or None if the LLM call fails.
    """
    try:
        prompt = _CLASSIFY_PROMPT.format(question=question[:300])
        resp = get_client().chat.completions.create(
            model=get_fast_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        if answer.startswith("YES"):
            return True
        if answer.startswith("NO"):
            return False
        # Unrecognised response – fall through to heuristic
        logger.debug("LLM image-relevance returned unexpected: %s", answer)
        return None
    except Exception as exc:
        logger.warning("LLM image-relevance classification failed: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────

def should_show_images(question: str) -> bool:
    """Decide whether the given question warrants accompanying images.

    Uses a fast LLM call first, with a keyword-based fallback.
    """
    if not question or not question.strip():
        return False

    # Try LLM classification first (fast model, ~100-200ms)
    llm_result = _llm_classify(question)
    if llm_result is not None:
        logger.info(
            "Image relevance for '%.60s': %s (LLM)",
            question, "show" if llm_result else "skip",
        )
        return llm_result

    # Fallback to keyword heuristics
    result = _keyword_heuristic(question)
    logger.info(
        "Image relevance for '%.60s': %s (heuristic fallback)",
        question, "show" if result else "skip",
    )
    return result