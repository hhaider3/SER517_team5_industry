"""
Self-assessment evaluation pipeline for the Django app.

This mirrors the General AI evaluation logic:
- Takes self-assessment JSON (problem + knowledge_types)
- Builds a combined student_text
- For each dimension (Facts/Strategies/Procedures/Rationales), constructs
  a DIMENSION_PROMPT with KB highlights, RAG snippets, few-shot examples,
  and CoT checklist.
- Calls OpenAI to get a labeled report, parses it into a structured form,
  and computes priority using priority_report.
"""

import logging
from typing import Any, Dict, List, Optional

from ava_apps.core.services.llm_client import get_client, get_model
from ava_apps.main.models import SelfAssessment
from ava_apps.core.services.knowledge_hub_service import collect_support_components
from ava_apps.self_assessment.services.priority_report_service import compute_full_priority_from_report


DIMENSION_PROMPT_TEMPLATE = """
You are a veteran diagnostician for {domain}/{branch} learning.
Your task: evaluate the student's self-assessment on the {dimension_name} dimension.

Use ONLY the resources below plus the student's own words. Never hallucinate new curriculum elements.

Knowledge Base Highlights:
{kb_highlights}

RAG Evidence:
{rag_evidence}

Few-Shot Calibrators:
{few_shot_block}

Chain-of-Thought Checklist:
{cot_checklist}

Label glossary (always reference explicitly):
- Know-Know: student stated something relevant and correct.
- Know-Don't Know: student openly noted a relevant gap to address.
- False Knowledge: student asserted something relevant but wrong.
- Omission: student skipped a required idea.
- Irrelevant Knowledge: off-topic information that should be redirected.

STUDENT SELF-ASSESSMENT:
{student_text}

WORKFLOW (MUST follow all 3 phases, every time):
<scratchpad>
PHASE 1 – Reference Blueprint
- Derive the ideal knowledge set for {dimension_name} using the highlights/RAG/few-shot exemplars.
- List concrete checkpoints (facts, strategies, steps, rationales) the learner should demonstrate.

PHASE 2 – Student Comparison
- Map the student's statements onto the blueprint.
- Note evidence of strengths, explicit uncertainties, wrong claims, omissions, or irrelevant tangents.
- Decide labels from [Know-Know, Know-Don't Know, False Knowledge, Omission, Irrelevant Knowledge] for each aspect.

PHASE 3 – Output Plan
- Choose the highest-leverage aspects to surface in the public report.
- Draft explanations that cite BOTH the student's wording and the reference blueprint.
</scratchpad>

OUTPUT EXACTLY:
<final>
Title: {dimension_name} Dimension
- Aspect: <short name>
  Labels: [comma-separated labels]
  Explanation: multi-sentence guidance referencing the student's wording and the knowledge base
(repeat for each aspect, including omissions or misconceptions)
Most critical gap: <one sentence naming the highest priority fix>
</final>
"""


DIMENSION_NAMES = ["Facts", "Strategies", "Procedures", "Rationales"]


def create_self_assessment_text(assessment: Dict[str, Any]) -> str:
    """
    Convert a structured self-assessment payload into plain text for evaluation.
    Mirrors General AI's create_self_assessment_text.
    """
    sa = assessment.get("self_assessment", {})
    se = assessment.get("self_evaluation", {})
    text_parts: List[str] = []

    problem = sa.get("problem", "")
    if problem:
        text_parts.append(f"Example Problem: {problem}")

    for kt in sa.get("knowledge_types", []):
        k_type = kt.get("type", "Unknown").capitalize()
        text_parts.append(f"{k_type}:")

        examples = kt.get("examples", {})
        if isinstance(examples, dict):
            for key, value in examples.items():
                text_parts.append(f"  - {key}: {value}")
        elif isinstance(examples, str):
            text_parts.append(f"  - Example: {examples}")

        uncertainties = kt.get("uncertainties", "")
        if uncertainties:
            text_parts.append(f"  - Uncertainties: {uncertainties}")
        text_parts.append("")

    if se:
        text_parts.append("Self-Evaluation:")
        for key, value in se.items():
            text_parts.append(f"  - {str(key).capitalize()}: {value}")

    return "\n".join(part for part in text_parts if part.strip())


def _extract_public_response(text: str) -> str:
    """Internal helper to handle extract public response."""
    if not text:
        return ""
    start_tag = "<final>"
    end_tag = "</final>"
    start = text.find(start_tag)
    end = text.find(end_tag, start + len(start_tag)) if start != -1 else -1
    if start != -1 and end != -1:
        return text[start + len(start_tag) : end].strip()
    return text.strip()


def _parse_dimension_output(name: str, text: str) -> Optional[Dict[str, Any]]:
    """
    Parse the DIMENSION_PROMPT output into a structured dict.
    Direct port of GeneralSubjectModule._parse_dimension_output.
    """
    if not text:
        return None
    title = None
    aspects: List[Dict[str, Any]] = []
    gap = None
    current: Optional[Dict[str, Any]] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Title:"):
            title = line.split(":", 1)[1].strip() or title
            continue
        if line.startswith("- Aspect:"):
            aspect_name = line.split(":", 1)[1].strip()
            current = {"aspect": aspect_name, "labels": [], "explanation": ""}
            aspects.append(current)
            continue
        if line.startswith("Labels:"):
            labels_str = line.split(":", 1)[1].strip()
            labels_str = labels_str.strip("[]")
            if current is not None:
                labels = [label.strip() for label in labels_str.split(",") if label.strip()]
                current["labels"] = labels
            continue
        if line.startswith("Explanation:"):
            explanation = line.split(":", 1)[1].strip()
            if current is not None:
                current["explanation"] = explanation
            continue
        if line.lower().startswith("most critical"):
            gap = line.split(":", 1)[1].strip()
            continue
        if current is not None:
            existing = current.get("explanation", "")
            current["explanation"] = (existing + " " + line).strip()

    if not aspects:
        return None
    return {
        "title": title or f"{name} Dimension",
        "aspects": aspects,
        "most_critical_gap": gap or "",
    }


def evaluate_self_assessment(student_id: str, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline:
    - For each dimension, build DIMENSION_PROMPT with KB support.
    - Call OpenAI, parse <final> block, accumulate structured report.
    - Compute priority from structured report.
    """
    preference_meta = assessment_data.get("preference_meta") or {}
    domain = preference_meta.get("domain") or "general-learning"
    branch = preference_meta.get("branch") or "exploratory"

    student_text = create_self_assessment_text(assessment_data)
    results: List[str] = []
    structured_dimensions: Dict[str, Dict[str, Any]] = {}

    for name in DIMENSION_NAMES:
        support = collect_support_components(
            subject=domain,
            dimension=name,
            query=student_text,
            preference_meta=preference_meta,
        )
        kb_highlights = "\n".join(f"- {item}" for item in support["highlights"]) or "- (none)"
        rag_evidence = "\n".join(f"- {hit}" for hit in support["rag_hits"]) or "- (none)"

        prompt = DIMENSION_PROMPT_TEMPLATE.format(
            domain=domain,
            branch=branch,
            dimension_name=name,
            kb_highlights=kb_highlights,
            rag_evidence=rag_evidence,
            few_shot_block=support["few_shot_text"],
            cot_checklist=support["cot_text"],
            student_text=student_text,
        )

        # Print prompt for visibility in terminal/VS Code when running self-assessment
        try:
            print("\n" + "=" * 100)
            print(f"‼️‼️Self-Assessment Input‼️‼️")
            print(f"[SelfAssessment] Dimension: {name}")
            print(f"[SelfAssessment] Domain/Branch: {domain}/{branch}")
            print(f"[SelfAssessment] Prompt:\n{prompt}")
            print("=" * 100 + "\n")
        except Exception:
            pass

        try:
            response = get_client().chat.completions.create(
                model=get_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1800,
            )
            raw_output = response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("Error during self-assessment LLM call for %s: %s", name, exc)
            raw_output = ""

        final_output = _extract_public_response(raw_output)
        results.append(f"--- {name} Dimension ---\n{final_output}\n")
        parsed_dimension = _parse_dimension_output(name, final_output)
        if parsed_dimension:
            structured_dimensions[name] = parsed_dimension

    evaluation_report = "\n\n".join(results)
    priority_result = compute_full_priority_from_report(structured_dimensions)

    return {
        "report": evaluation_report,
        "student_text": student_text,
        "priority": priority_result,
        "structured_report": structured_dimensions,
        "domain": domain,
        "branch": branch,
    }


def get_latest_evaluation_for_user(username: str, learning_goal_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch the most recent self-assessment evaluation for a given user
    from the self_assessments table, if it exists.
    """
    try:
        queryset = SelfAssessment.objects.filter(username=username)
        if learning_goal_id is not None:
            queryset = queryset.filter(learning_goal_id=int(learning_goal_id))
        latest = queryset.order_by("-created_at").first()
        if not latest:
            return None
        return {
            "id": latest.id,
            "username": latest.username,
            "learning_goal_id": latest.learning_goal_id,
            "domain": latest.domain,
            "branch": latest.branch,
            "student_text": latest.student_text,
            "evaluation_report": latest.evaluation_report,
            "structured_report": latest.structured_report,
            "priority": latest.priority,
            "created_at": latest.created_at,
        }
    except Exception as exc:  # pragma: no cover - defensive
        logging.error(f"Error fetching latest self-assessment for {username} (goal={learning_goal_id}): {exc}")
        return None


__all__ = [
    "evaluate_self_assessment",
    "create_self_assessment_text",
    "get_latest_evaluation_for_user",
]