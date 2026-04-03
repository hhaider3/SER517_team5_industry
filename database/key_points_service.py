from __future__ import annotations

import json
import logging
import re
from typing import List

from ava_apps.core.services.llm_client import get_client, get_model
from ava_apps.chat.general_chat.text_service import strip_html


logger = logging.getLogger(__name__)


def generate_key_points(answer_html: str, question: str = "") -> List[str]:
    """Generate key points."""
    plain = strip_html(answer_html)
    if not plain:
        return []

    question = (question or "").strip()
    prompt = (
        "You are extracting key points from a tutor's answer to check a learner's understanding.\n"
        "Only keep points that directly answer the learner's question. Ignore chit-chat, greetings, fluff,\n"
        "or tangential examples/metaphors that are not needed to answer the question.\n"
        "Return STRICT JSON: {\"points\": [\"...\"]} (no extra text).\n"
        "Rules:\n"
        "- Make each point concise but specific (name the fact/step/idea).\n"
        "- Remove duplicates/near-duplicates.\n"
        "- Focus on the core content that resolves the question; drop side remarks and minor asides.\n\n"
        f"Question:\n{question[:400] if question else '(not provided)'}\n\n"
        f"Tutor answer:\n{plain[:3000]}"
    )
    try:
        res = get_client().chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        content = res.choices[0].message.content or "{}"
        data = json.loads(content)
        if isinstance(data, dict):
            data_list = data.get("points") or []
        elif isinstance(data, list):
            data_list = data
        else:
            data_list = []
        seen = set()
        points: List[str] = []
        for x in data_list:
            s = str(x).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            points.append(s)
        return points
    except Exception as exc:
        logger.error("Key point extraction failed: %s", exc)

    parts = re.split(r"[.;]\s+", plain)
    seen = set()
    out: List[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
