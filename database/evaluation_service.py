from __future__ import annotations

import json
import logging
from typing import List, Tuple

from ava_apps.core.services.llm_client import get_client, get_model
from ava_apps.chat.general_chat.text_service import truncate_for_prompt


logger = logging.getLogger(__name__)


def evaluate_understanding_with_reasons(
    key_points: List[str],
    question: str,
    answer_html: str,
    user_summary: str,
) -> Tuple[List[str], List[str], List[dict], str]:
    """Handle evaluate understanding with reasons."""
    if not key_points:
        return [], [], [], ""

    understood: List[str] = []
    details: List[dict] = []
    prompt = (
        "You are evaluating a student's understanding of a tutor's answer. "
        "For EACH key point, decide if the student's summary sufficiently covers it. "
        "Output STRICT JSON only in this shape:\n"
        "{\n"
        '  "points": [ {"point": "…", "status": "PASS"|"FAIL", "reason": "short justification"} ],\n'
        '  "remediation": "If any FAIL, give a brief next-step explanation tailored to the student. '
        'If all PASS, use an empty string."\n'
        "}\n"
        "Rules: minor paraphrases count as PASS; stylistic differences are fine. "
        "Pass a point if the student restates the core idea even if they omit secondary effects/benefits; "
        "focus on the main clause that answers the question, ignore side benefits unless they are the core of the point. "
        "If unclear, be moderately forgiving and only FAIL when the core idea is missing or contradicted. "
        "Do not invent new key points.\n"
        "Few-shot scoring examples (treat them as binding):\n"
        "- Key: \"Use personal/real examples to make the answer persuasive.\" Student: \"Use my own experiences.\" -> PASS (core captured, benefit omitted).\n"
        "- Key: \"Explain photosynthesis uses sunlight to make glucose.\" Student: \"Plants use sunlight for food.\" -> PASS (core captured, detail omitted).\n"
        "- Key: \"State Pythagoras theorem (a^2+b^2=c^2).\" Student: \"Triangles have three sides.\" -> FAIL (core missing).\n\n"
        f"Here is the original student's question for your reference: {truncate_for_prompt(question, 400)}\n"
        f"Here is the tutor original answer for your reference: {truncate_for_prompt(answer_html, 800)}\n"
        f"Here is the Key Points needs to check one by one: {key_points}\n"
        f"Here is the student's answer needs to be check:\n{user_summary[:3000]}"
    )

    remediation_text = ""
    try:
        res = get_client().chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        content = res.choices[0].message.content or "{}"
        payload = json.loads(content)
        point_items = payload.get("points", []) if isinstance(payload, dict) else []
        remediation_text = str(payload.get("remediation", "")) if isinstance(payload, dict) else ""

        for item in point_items:
            point = str(item.get("point", "")).strip()
            status = str(item.get("status", "")).upper()
            reason = str(item.get("reason", "")).strip()
            if not point:
                continue
            if status == "PASS":
                understood.append(point)
            details.append(
                {
                    "point": point,
                    "status": "PASS" if status == "PASS" else "FAIL",
                    "reason": reason or ("Marked as PASS" if status == "PASS" else "Marked as FAIL"),
                }
            )

        if not details:
            for point in key_points:
                details.append({"point": point, "status": "FAIL", "reason": "Not found in student summary."})

        covered = {d.get("point", "") for d in details}
        for kp in key_points:
            if kp not in covered:
                status = "PASS" if kp in understood else "FAIL"
                details.append(
                    {
                        "point": kp,
                        "status": status,
                        "reason": "Added for completeness; not returned by model.",
                    }
                )

    except Exception as exc:
        logger.error("Understanding analysis failed: %s", exc)
        for point in key_points:
            details.append({"point": point, "status": "FAIL", "reason": "Analysis fallback: not matched."})

    unique_details: List[dict] = []
    seen_points = set()
    for d in details:
        pt = d.get("point")
        if pt in seen_points:
            continue
        seen_points.add(pt)
        unique_details.append(d)

    missing = [d["point"] for d in unique_details if d.get("status") != "PASS"]
    understood_norm = [d["point"] for d in unique_details if d.get("status") == "PASS"]
    return understood_norm, missing, unique_details, remediation_text
