"""
Dynamic self-assessment template generator.

Ports the JSON-emitting prompt from the standalone Dynamic Template project
so we can generate an example self-assessment text (and optional template)
for any domain/branch inside the Flask app.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ava_apps.core.services.llm_client import get_client, get_fast_model

# Prompts copied from the Dynamic Template service. Keep in sync if upstream changes.
SYSTEM_PROMPT = (
    "You are an AI that emits ONLY JSON per the schema:\n"
    "{\n"
    " 'fspr': {\n   'facts': [string],\n   'strategies': [string],\n   'procedures': [string],\n   'rationales': [string],\n   'notes': string?\n },\n"
    " 'template': {\n   'title': string,\n   'sections': [\n     {'heading': string, 'prompt': string, 'placeholders': [string], 'meta': {'experimental': boolean?}}\n   ],\n   'version': int,\n   'meta': {'changelog': [string]}\n },\n"
    " 'example_text': string,\n"
    " 'empty_template': {\n   'title': string,\n   'sections': [\n     {'heading': string, 'prompt': string (empty), 'placeholders': [] (empty), 'meta': {}}\n   ],\n   'version': int,\n   'meta': {}\n }\n"
    "}\n"
    "Rules:\n"
    "- No prose outside JSON. Output valid UTF-8 JSON only.\n"
    "- Keep bullets ≤ 30 words.\n"
    "- Each procedure begins with '1) ', '2) ', etc and contains a measurable outcome.\n"
    "- Generate 'example_text': A detailed, comprehensive self-assessment example (4-6 paragraphs, 500-800 words) for a CLOSELY RELATED topic to user_prompt - use a variant, related concept, or adjacent topic in the SAME specific domain.\n"
    "  Hard rule: the example MUST NOT be the exact same topic phrase as user_prompt. Do not reuse the exact topic phrase in [SIMILAR_TOPIC] or the opening sentence. Choose a sibling/adjacent subtopic instead.\n"
    "  For example, if user asks about '401K', use 'IRA' or 'Roth IRA' (retirement accounts, not just 'finance'). If user asks about 'cancer', use 'diabetes' or 'heart disease' (medical conditions, not just 'health'). Make it SPECIFIC to the user's domain.\n"
    "  Topic-shift few-shot examples (input -> SIMILAR_TOPIC):\n"
    "   - linear algebra -> eigenvalues and eigenvectors\n"
    "   - calculus derivatives -> limits and continuity\n"
    "   - photosynthesis -> cellular respiration\n"
    "  Structure:\n"
    "  1. Introduction: 'I want to teach you how to assess your own knowledge that you have about a subject area. Let's do this by taking an example that you already know. Suppose you wanted to assess your own knowledge about [SIMILAR_TOPIC]. If I want to be able to [DO_THIS], I need four types of knowledge. These are facts, strategies, procedures and rationales. [Explain each type with examples]. You can think of facts as telling you \"what\", strategies and procedures as telling you \"how\" and rationales as telling you \"why\".'\n"
    "  2. Detailed Facts section: 'For facts, I need to know [list]. [Explain each fact in detail with definitions].'\n"
    "  3. Detailed Strategies section: 'For strategies, I need to know [strategy]. [Explain strategy in detail, including any uncertainties].'\n"
    "  4. Detailed Procedures section: 'For procedures, I need to know [procedure]. [Explain each procedure step-by-step, including any uncertainties].'\n"
    "  5. Detailed Rationales section: 'For rationales, I believe [rationale]. [Explain why each rationale matters].'\n"
    "  6. Self-reflection: 'When I look over what I wrote, I see that [self-assessment of strengths and gaps for each category].'\n"
    "- Generate 'empty_template': A template with empty prompts/placeholders for the user's ACTUAL topic (from user_prompt). Keep headings but blank prompts/placeholders.\n"
    "- Evolve the template gently using up to K prior templates: preserve high-signal sections, retire low-signal ones, add at most one experimental section marked meta.experimental=true."
)

SELF_ASSESSMENT_GUIDE = (
    "Self-assessment framework for mapping any topic to FSPR (facts, strategies, procedures, rationales).\n"
    "Use these rules when producing FSPR so outputs track the user's domain.\n"
    "IMPORTANT: When generating example_text, choose a CLOSELY RELATED topic in the SAME specific domain as user_prompt. "
    "Hard rule: do not reuse the exact user_prompt topic phrase for SIMILAR_TOPIC; pick an adjacent subtopic instead. "
    "Be specific: if user asks about '401K', use 'IRA' or 'Roth 401K' (retirement accounts), not general 'finance'. "
    "If user asks about 'cancer', use 'diabetes' or 'heart disease' (medical conditions), not general 'health'.\n\n"
    "HISTORY example (Declaration of Independence):\n"
    "- Facts: key people, dates, locations, context (e.g., Jefferson, 1776, King George III, British rule).\n"
    "- Strategies: problem-solution framing (colonists oppose British rule; declare independence; seek allies).\n"
    "- Procedures: sequence of events (protests → war → declaration → French alliance → victory).\n"
    "- Rationales: why events occurred and impacts (convince France; birth of democracy; global effects).\n\n"
    "LITERATURE example (Little Red Riding Hood):\n"
    "- Facts: characters, setting, time.\n"
    "- Strategies: plot structure to express theme (warn about strangers).\n"
    "- Procedures: concrete scenes in order (errand → wolf trick → rescue).\n"
    "- Rationales: author purpose, character goals, how elements reinforce theme.\n\n"
    "MATH example (solve 2-step equations):\n"
    "- Facts: variables, constants, coefficients, expressions, equations.\n"
    "- Strategies: reverse order of operations (SADMEP).\n"
    "- Procedures: additive/multiplicative inverses, properties of equality.\n"
    "- Rationales: subtraction/division property of equality justify steps.\n\n"
    "Always fit the current user_prompt to this mapping."
)

FALLBACK_EXAMPLE_TEMPLATE = """
I want to teach you how to assess your own knowledge that you have about a subject area. Let's do this by taking an example that you already know. Suppose you wanted to assess your own knowledge about {topic}. If I want to be able to get better at this, I need four types of knowledge. These are facts, strategies, procedures and rationales. Facts tell me what the ideas are, strategies and procedures tell me how to act on them, and rationales tell me why the steps make sense. With this in mind, this is how I might assess my own knowledge of {topic}.

For facts, I list the essential concepts, ingredients and vocabulary for {topic}. I write out brief definitions for each one so I know whether I truly understand them. If I see missing terms or fuzzy explanations, that shows me where I need to review core knowledge before moving on.

For strategies, I outline the main approaches I should use when working within {topic}. These are the playbooks or plans that guide my decisions. I check whether I know when to use each strategy, where I hesitate, and which parts of the approach are unclear so I can tighten them up.

For procedures, I break strategies into repeatable steps that I could hand to someone else. I check each step for the exact action, any ordering rules, and the expected result. Any step where I am unsure about prerequisites, inputs, or how to verify correctness becomes an area to study next.

For rationales, I write down the reasons the strategies and procedures work. This includes any principles, properties, or cause-and-effect relationships that justify the steps. If I cannot explain why a step is necessary, I know I need to revisit the underlying ideas so I am not just memorizing motions.

When I look over what I wrote, I can see my strengths and gaps for facts, strategies, procedures and rationales around {topic}. Solid definitions suggest my foundation is good. Unclear decision points or missing steps show where I should practice. Any fragile rationale signals that I should dig deeper into the principles so my knowledge is reliable and transferable.
"""


def _build_topic(domain: str, branch: str) -> str:
    """Internal helper to build topic."""
    parts = [p for p in [domain, branch] if p]
    return " / ".join(parts) if parts else "general learning"


def _fallback_example_text(topic: str) -> str:
    """Internal helper to handle fallback example text."""
    return FALLBACK_EXAMPLE_TEMPLATE.format(topic=topic or "general learning").strip()


def generate_dynamic_example_text(
    domain: str = "",
    branch: str = "",
    preference: str = "",
    prior_templates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Invoke the Dynamic Template prompt to get an example_text and empty_template.
    Returns a dict with example_text (never empty) plus template/fspr payloads.
    """
    topic = _build_topic(domain, branch)
    user_prompt = f"Self-assessment template for {topic}"
    if preference:
        user_prompt += f" with learner goal: {preference}"

    try:
        print(
            f"[SelfAssessment] Dynamic example generation start topic={topic}",
            flush=True,
        )
    except Exception:
        pass

    user_payload = json.dumps(
        {"user_prompt": user_prompt, "prior_templates": prior_templates or []},
        ensure_ascii=False,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": SELF_ASSESSMENT_GUIDE},
        {"role": "user", "content": user_payload},
    ]

    try:
        response = get_client().chat.completions.create(
            model=get_fast_model(),
            messages=messages,
            temperature=0.3,
            timeout=60,
        )
        raw_content = response.choices[0].message.content or "{}"
        payload = json.loads(raw_content)
        example_text = (payload.get("example_text") or "").strip()
        used_fallback = False
        if not example_text:
            example_text = _fallback_example_text(topic)
            used_fallback = True
        try:
            print(
                "[SelfAssessment] Dynamic example generation done "
                f"(fallback={used_fallback}, length={len(example_text)})",
                flush=True,
            )
        except Exception:
            pass
        return {
            "example_text": example_text,
            "template": payload.get("empty_template") or payload.get("template"),
            "fspr": payload.get("fspr"),
            "raw_payload": payload,
        }
    except Exception as exc:  # pragma: no cover - defensive
        logging.error("Dynamic template generation failed for %s: %s", topic, exc)
        try:
            fallback_text = _fallback_example_text(topic)
            print(
                "[SelfAssessment] Dynamic example generation failed, "
                f"using fallback (length={len(fallback_text)})",
                flush=True,
            )
        except Exception:
            fallback_text = _fallback_example_text(topic)
        return {
            "example_text": fallback_text,
            "template": None,
            "fspr": None,
            "raw_payload": None,
        }


__all__ = ["generate_dynamic_example_text"]