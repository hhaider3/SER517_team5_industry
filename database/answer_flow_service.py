from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

from markdown import Markdown
from pymdownx.arithmatex import makeExtension

from ava_apps.chat.shared import conversation_memory_service
from ava_apps.core.services.database_service import db_service
from ava_apps.learning_goal.services import learning_goal_service
from ava_apps.chat.general_chat.services.answer_generation_service import get_ai_answer
from ava_apps.chat.general_chat.services.image_search_service import search_images, build_image_gallery_html
from ava_apps.chat.general_chat.services.wikimedia_image_service import search_wikimedia_images, build_wikimedia_gallery_html
from ava_apps.chat.general_chat.services.image_relevance_service import should_show_images
from ava_apps.self_assessment.services.self_assessment_evaluation_service import get_latest_evaluation_for_user

from ..check_understanding.key_points_service import generate_key_points
from .text_service import detect_answer_style, fix_math_expressions, strip_html


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlowResponse:
    ok: bool
    status: int = 200
    payload: Dict[str, Any] = field(default_factory=dict)


def _build_pending_reminder_html(missing_list: list[Any]) -> str:
    """Internal helper to build pending reminder html."""
    bullets = "".join([f"<li>{strip_html(str(p))}</li>" for p in missing_list]) or "<li>(no points listed)</li>"
    return (
        "<div class=\"alert alert-warning\" role=\"alert\">"
        "Please finish restating the missing points before asking a new question:"
        f"<ul>{bullets}</ul>"
        "</div>"
    )


def answer_question_flow(
    username: str,
    user_profile: Dict[str, Any],
    question: str,
    learning_goal_id: int,
    keyword_synonyms: Dict[str, Any],
    augment_weight_from_answer: bool = False,
) -> FlowResponse:
    """Handle answer question flow."""
    goal = learning_goal_service.get_learning_goal(username, learning_goal_id)
    if not goal:
        return FlowResponse(ok=False, status=404, payload={"error": "Learning goal not found."})
    if (goal.get("status") or "").strip() != "self_assessment_completed":
        return FlowResponse(
            ok=False,
            status=400,
            payload={"error": "Self-assessment not completed for this learning goal."},
        )

    if not question:
        return FlowResponse(ok=False, status=400, payload={"error": "Question cannot be empty."})
    if not isinstance(question, str) or len(question) > 500:
        return FlowResponse(ok=False, status=400, payload={"error": "Invalid question."})
    if not user_profile:
        return FlowResponse(ok=False, status=404, payload={"error": "User profile not found."})

    try:
        user_profile["preference_meta"] = {
            "domain": (goal.get("domain") or "general-learning"),
            "branch": (goal.get("branch") or "exploratory"),
            "preference": (goal.get("preference_text") or goal.get("title") or ""),
        }

        latest_eval = get_latest_evaluation_for_user(username, learning_goal_id=learning_goal_id)
        if latest_eval:
            domain_from_eval = (latest_eval.get("domain") or "").strip()
            branch_from_eval = (latest_eval.get("branch") or "").strip()
            if domain_from_eval:
                user_profile["preference_meta"]["domain"] = domain_from_eval
            if branch_from_eval:
                user_profile["preference_meta"]["branch"] = branch_from_eval

            if latest_eval.get("evaluation_report"):
                user_profile["evaluation_report"] = latest_eval.get("evaluation_report")
                priority_raw = latest_eval.get("priority")
                priority_dict = None
                if isinstance(priority_raw, str):
                    try:
                        priority_dict = json.loads(priority_raw)
                    except json.JSONDecodeError:
                        priority_dict = None
                elif isinstance(priority_raw, dict):
                    priority_dict = priority_raw

                if priority_dict:
                    top_dim = priority_dict.get("top_dimension")
                    top_label = priority_dict.get("top_label_in_top_dimension")
                    if top_dim and top_label:
                        user_profile["priority_focus"] = f"{top_dim} – {top_label}"
                    user_profile["evaluation_priority"] = priority_dict
    except Exception as exc:
        logger.error("Error attaching latest evaluation for user %s (goal=%s): %s", username, learning_goal_id, exc)

    pending_remediation = learning_goal_service.get_goal_pending_remediation(username, learning_goal_id) or {}
    remediation_kwargs = {}
    if pending_remediation:
        remediation_kwargs = {
            "remediation_mode": True,
            "original_answer": pending_remediation.get("original_answer"),
            "original_answer_style": pending_remediation.get("original_answer_style"),
            "missed_facts": pending_remediation.get("missed_facts"),
            "wrong_facts": pending_remediation.get("wrong_facts"),
            "recall_score": pending_remediation.get("recall_score"),
            "attempt_number": pending_remediation.get("attempt_number", 1),
            "remediation_question": pending_remediation.get("original_question"),
            "user_summary": pending_remediation.get("user_summary"),
            "remediation_report": pending_remediation.get("remediation_report"),
        }

    if pending_remediation.get("awaiting_understanding"):
        if not pending_remediation.get("key_points"):
            key_points_now = generate_key_points(
                pending_remediation.get("original_answer", ""),
                pending_remediation.get("original_question", ""),
            )
            pending_remediation["key_points"] = key_points_now
            pending_remediation["missing_points"] = key_points_now
            learning_goal_service.set_goal_pending_remediation(username, learning_goal_id, pending_remediation)

        missing_list = pending_remediation.get("missing_points") or pending_remediation.get("key_points") or []
        reminder_html = _build_pending_reminder_html(missing_list)
        return FlowResponse(
            ok=True,
            status=200,
            payload={
                "answer": reminder_html,
                "answer_style": "informational",
                "history_id": None,
                "question": question,
            },
        )

    conversation_id = learning_goal_service.ensure_goal_conversation_id(username, learning_goal_id)
    if not conversation_id:
        return FlowResponse(
            ok=False,
            status=500,
            payload={"error": "Could not create or load remote conversation for this learning goal."},
        )

    answer = get_ai_answer(
        question,
        user_profile,
        keyword_synonyms,
        conversation_id=conversation_id,
        **remediation_kwargs,
    )
    conversation_memory_service.maybe_compact_conversation_id(conversation_id)

    if not answer:
        return FlowResponse(
            ok=False,
            status=500,
            payload={"error": "Could not generate an answer. Please try again later."},
        )

    answer = fix_math_expressions(answer)
    md = Markdown(extensions=["extra", makeExtension(generic=True)])
    answer_html = md.convert(answer)

    # --- Image search integration (Wikimedia → ChromaDB fallback) ---
    # Only fetch images when the question actually benefits from visuals
    if should_show_images(question):
        try:
            images = search_wikimedia_images(query_text=question, n_results=3)
            gallery_html = build_wikimedia_gallery_html(images) if images else ""
            if not gallery_html:
                # Fallback to local ChromaDB if Wikimedia returns nothing
                local_images = search_images(query_text=question, n_results=3)
                gallery_html = build_image_gallery_html(local_images)
            if gallery_html:
                answer_html += gallery_html
        except Exception as exc:
            logger.warning("Image search skipped for question '%s': %s", question[:60], exc)
    else:
        logger.info("Images skipped (not relevant) for question: '%.60s'", question)

    detected_answer_style = detect_answer_style(answer)
    history_id = db_service.add_user_history(
        username,
        question,
        answer_html,
        detected_answer_style,
        learning_goal_id=learning_goal_id,
    )

    pending_payload = {
        "original_question": question,
        "original_answer": answer_html,
        "original_answer_style": detected_answer_style,
        "history_id": history_id,
        "key_points": [],
        "missing_points": [],
        "awaiting_understanding": True,
        "attempt_number": 1,
        "remediation_report": None,
    }
    learning_goal_service.set_goal_pending_remediation(username, learning_goal_id, pending_payload)

    if augment_weight_from_answer:
        fresh_profile = db_service.get_user_by_username(username) or {}
        weight_fields = ["informational", "real_world", "cause_and_effect", "goal_based"]
        weight_words = ["informational", "real world", "cause and effect", "goal based"]
        updated_weights = fresh_profile.get("answer_type_weights", {}).copy()
        weights_updated = False
        for i, weight_field in enumerate(weight_fields):
            if weight_words[i] in answer_html.lower() and updated_weights.get(weight_field, 0.5) <= 0.9:
                updated_weights[weight_field] += 0.1
                weights_updated = True
        if weights_updated:
            db_service.update_user(username, {"answer_type_weights": updated_weights})

    return FlowResponse(
        ok=True,
        status=200,
        payload={
            "answer": answer_html,
            "answer_style": detected_answer_style,
            "history_id": history_id,
            "question": question,
        },
    )