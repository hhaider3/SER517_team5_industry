# services/answer_service.py
"""
Adaptive Virtual Assistant Answer Service

AI-powered answer generation with personalized responses based on user profiles
and adaptive learning algorithms. Integrates with Open AI's model.

Key Features:
- Adaptive answer style selection using 80-20 exploration/exploitation
- Question type classification (Fact, Procedural, Rationale)
- User profile-based personalization
- AI-powered understanding assessment
"""

import logging
import random
import string
import os
from datetime import datetime
from typing import List, Optional, TypedDict
import re
from ava_apps.core.services.llm_client import get_client, get_model
from ava_apps.core.services.knowledge_hub_service import collect_support_components


# Remediation Answer Generation - Interface Definitions
class RemediationAnswerInput(TypedDict, total=False):
    """
    Input interface for remediation answer generation.

    Required fields:
        question (str): The original question being re-answered
        user_profile (dict): User's complete profile with learning preferences

    Optional fields (for remediation mode):
        remediation_mode (bool): Whether this is a remediation attempt (default: False)
        original_answer (str): The original answer provided to the user
        original_answer_style (str): Style used in original answer (informational, real_world, etc.)
        missed_facts (List[str]): Facts from original answer that user failed to recall
        wrong_facts (List[str]): Facts user recalled incorrectly
        recall_score (float): User's recall score (0-100)
        attempt_number (int): Which remediation attempt this is (1, 2, 3...)
        keyword_synonyms (dict): Keyword mapping for context enhancement
        remediation_report (str): Full remediation report text from the previous turn (optional)
    """

    question: str
    user_profile: dict
    remediation_mode: bool
    original_answer: Optional[str]
    original_answer_style: Optional[str]
    missed_facts: Optional[List[str]]
    wrong_facts: Optional[List[str]]
    recall_score: Optional[float]
    attempt_number: Optional[int]
    keyword_synonyms: Optional[dict]
    remediation_report: Optional[str]


class RemediationAnswerOutput(TypedDict):
    """
    Output interface for remediation answer generation.

    Fields:
        answer (str): The generated answer text (markdown/HTML format)
        answer_style (str): The answer style used (informational, real_world, cause_and_effect, goal_based)
        question_type (str): Detected question type (Fact, Procedural, Rationale, Unknown)
        difficulty_level (str): Difficulty level (beginner, intermediate, advanced)
        target_fact_count (int): Target number of facts in the answer
        is_remediation (bool): Whether this was a remediation answer
        remediation_strategy (Optional[str]): Strategy used for remediation (if applicable)
    """

    answer: str
    answer_style: str
    question_type: str
    difficulty_level: str
    target_fact_count: int
    is_remediation: bool
    remediation_strategy: Optional[str]


def truncate_text(value: Optional[str], limit: int = 800) -> str:
    """
    Keep long strings compact inside prompts to avoid overwhelming the model.
    """
    if not value:
        return "N/A"
    value = str(value).strip()
    return value if len(value) <= limit else value[:limit] + "..."


def process_question_keywords(question, keyword_synonyms):
    """
    Extract and match keywords from user questions using synonym mapping.

    Args:
        question (str): The user's input question
        keyword_synonyms (dict): Dictionary mapping keywords to their synonyms

    Returns:
        list: Matched keywords found in the question, or all words if no matches
    """
    words = question.split()
    return [word for word in words if word.lower() in keyword_synonyms] or words


def detect_question_type(question):
    """
    Classify user questions into educational categories using keyword analysis.

    Classification Logic:
    - Fact: Questions seeking specific information (what, when, where, who)
    - Procedural: Questions about processes or methods (how, steps, process)
    - Rationale: Questions seeking explanations or reasoning (why, reason, explain)
    - Unknown: Questions that don't fit the above categories

    Args:
        question (str): The user's input question

    Returns:
        str: Question type classification ("Fact", "Procedural", "Rationale", "Unknown")
    """
    q = question.lower()

    # Check for different question type indicators
    if any(term in q for term in ["what", "when", "where", "who"]):
        return "Fact"
    elif any(term in q for term in ["how", "steps", "process"]):
        return "Procedural"
    elif any(term in q for term in ["why", "reason", "explain"]):
        return "Rationale"

    return "Unknown"


# Ka$h-new
def classify_question_difficulty(question: str, user_profile: dict) -> str:
    """
    Classify the difficulty of a question as Beginner, Intermediate, or Advanced.

    Uses heuristics including:
    - Keyword and phrasing complexity
    - Length and structure
    - Historical user interactions

    Args:
        question (str): The input question from the user.
        user_profile (dict): User profile containing interaction history and level markers.

    Returns:
        str: One of "beginner", "intermediate", or "advanced"
    """

    # Basic text prep
    q = question.lower().translate(str.maketrans("", "", string.punctuation))
    words = q.split()
    word_count = len(words)

    # Keyword-based signals
    beginner_keywords = {
        "define",
        "what is",
        "examples of",
        "simple",
        "basic",
        "explain like",
        "what are",
        "list",
        "identify",
        "name",
        "describe",
        "can you tell me",
        "introduction to",
        "overview of",
        "easy way to",
        "how to start",
        "give me an example",
        "show me",
        "illustrate",
        "such as",
        "what does [X] do",
        "how does [X] work (simply)",
        "meaning of",
        "what is the term for",
    }
    intermediate_keywords = {
        "difference between",
        "how does",
        "steps of",
        "purpose of",
        "compare",
        "how does [X] function",
        "process of",
        "mechanisms behind",
        "stages of",
        "contrast",
        "relation between",
        "connection between",
        "factors affecting",
        "role of",
        "when to use",
        "applications of",
        "how to apply",
        "practical use of",
        "why is [X] important",
        "reasons for",
        "explain the purpose",
        "significance of",
        "how to solve",
        "steps to approach",
        "strategies for",
        "what happens if",
        "effect of",
        "impact of",
    }
    advanced_keywords = {
        "evaluate",
        "derive",
        "prove",
        "analyze",
        "implications",
        "theory of",
        "limitations",
        "critique",
        "assess",
        "critically evaluate",
        "disadvantages of",
        "advantages of",
        "pros and cons of",
        "formulate",
        "deduce",
        "demonstrate",
        "establish",
        "show that",
        "model of",
        "framework for",
        "theoretical basis",
        "underlying principles",
        "design a system",
        "develop a methodology",
        "optimize",
        "implement a solution",
        "architecture of",
        "predict the outcome",
        "forecast the trends",
        "speculate on",
        "synthesize information",
        "integrate concepts",
        "combine elements",
        "research methods for",
        "experimental design",
        "data analysis techniques",
        "ethical considerations",
        "societal impact",
        "policy implications",
    }

    beginner_hits = sum(1 for kw in beginner_keywords if kw in q)
    intermediate_hits = sum(1 for kw in intermediate_keywords if kw in q)
    advanced_hits = sum(1 for kw in advanced_keywords if kw in q)

    # Heuristic rules
    if advanced_hits >= 1 or word_count > 20:
        return "advanced"
    elif intermediate_hits >= 1 or 12 < word_count <= 20:
        return "intermediate"
    else:
        return "beginner"


def detect_topic_label(question, keyword_synonyms):
    """
    Detect a topic label for the given question using keyword matching.

    Args:
        question (str): The user's question
        keyword_synonyms (dict): A dict mapping topic keys to lists of related keywords/synonyms

    Returns:
        str: Best-matching topic label, or "unknown_topic" if none found
    """
    question_lower = question.lower()
    best_match = "unknown_topic"
    max_hits = 0

    for topic, synonyms in keyword_synonyms.items():
        hits = sum(1 for word in synonyms if word in question_lower)
        if hits > max_hits:
            best_match = topic
            max_hits = hits

    return best_match


def determine_best_answer_style(user_profile):
    """
    Implement adaptive answer style selection using 80-20 exploration-exploitation algorithm.

    Balances between:
    - Exploitation (80%): Using the user's most successful answer style
    - Exploration (20%): Trying other styles to discover better preferences

    Args:
        user_profile (dict): User profile containing answer_type_weights

    Returns:
        str: Selected answer style for the current response
    """
    import random

    # Extract adaptive weights with defaults
    adaptive_weights = user_profile.get(
        "answer_type_weights",
        {
            "informational": 0.5,
            "real_world": 0.5,
            "cause_and_effect": 0.5,
            "goal_based": 0.5,
        },
    )

    # Apply 80-20 exploration-exploitation strategy
    if random.random() < 0.8:  # 80% exploitation
        return max(adaptive_weights.items(), key=lambda x: x[1])[0]
    else:  # 20% exploration
        # Exclude current best style from exploration
        highest_style = max(adaptive_weights.items(), key=lambda x: x[1])[0]
        exploration_weights = {
            k: v for k, v in adaptive_weights.items() if k != highest_style
        }

        # Weighted random selection from exploration pool
        total = sum(exploration_weights.values())
        if total == 0:
            return random.choice(list(exploration_weights.keys()))

        r = random.random() * total
        cumulative = 0
        for style, weight in exploration_weights.items():
            cumulative += weight
            if r <= cumulative:
                return style

        return list(exploration_weights.keys())[0]


# Remediation Prompt Builder
def build_remediation_prompt(
    question,
    user_profile,
    missed_facts,
    wrong_facts,
    recall_score,
    attempt_number,
    original_answer_style,
    preferred_answer_style,
    user_topic_level,
    difficulty_level,
    topic_label,
    original_answer=None,
    user_summary=None,
    remediation_report: Optional[str] = None,
):
    """
    Build a specialized prompt for remediation attempts that focuses on gaps and corrections.

    Remediation Strategy:
    - Focus on facts the user missed or got wrong
    - Use a different answer style than the original (to aid comprehension)
    - Adjust complexity based on recall score
    - Provide clear corrections for wrong facts
    - Use reinforcement for concepts user struggles with

    Args:
        question (str): Original question
        user_profile (dict): User profile
        missed_facts (List[str]): Facts user didn't recall
        wrong_facts (List[str]): Facts user recalled incorrectly
        recall_score (float): User's recall score (0-100)
        attempt_number (int): Which attempt this is
        original_answer_style (str): Style used in original answer
        preferred_answer_style (str): Style to use for remediation
        user_topic_level (str): User's level for this topic
        difficulty_level (str): Detected difficulty
        topic_label (str): Topic being covered
        original_answer (str): The original answer text from the tutor
        user_summary (str): The learner's recall/summary of the answer
        remediation_report (str): Full remediation report captured after the last turn

    Returns:
        str: Customized remediation prompt for AI
    """
    # Initialize default lists if None
    missed_facts = missed_facts or []
    wrong_facts = wrong_facts or []

    # Determine remediation approach based on recall score
    if recall_score < 40:
        remediation_approach = "fundamental_review"
        complexity_instruction = "significantly simplify the explanation, break into smaller steps, use analogies"
    elif recall_score < 70:
        remediation_approach = "targeted_gaps"
        complexity_instruction = "focus on the specific gaps, provide clearer examples"
    else:
        remediation_approach = "minor_clarification"
        complexity_instruction = "clarify the subtle points that were missed"

    # Build remediation-specific instructions
    original_answer_snippet = truncate_text(original_answer, 800)
    user_summary_snippet = truncate_text(user_summary, 400)
    remediation_report_snippet = (
        truncate_text(remediation_report, 600)
        if remediation_report
        else "Not provided; lean on the missed/wrong facts list."
    )

    remediation_context = f"""
    <REMEDIATION_MODE>
    This is a REMEDIATION attempt (attempt #{attempt_number}) for a question the user previously struggled with.

    LAST EXCHANGE SNAPSHOT:
    • Previous question: {question}
    • Tutor's prior answer (snippet): {original_answer_snippet}
    • Learner's recall/summary: {user_summary_snippet}
    • Original Style -> Current Style: {original_answer_style} -> {preferred_answer_style}
    • Remediation report: {remediation_report_snippet}

    KEY GAPS TO FIX NOW:
    """

    if missed_facts:
        remediation_context += (
            f"\n    • Missed facts (top {min(len(missed_facts),5)}): "
            + "; ".join(missed_facts[:5])
        )

    if wrong_facts:
        remediation_context += (
            f"\n    • Wrong facts (top {min(len(wrong_facts),5)}): "
            + "; ".join(wrong_facts[:5])
        )

    remediation_context += f"""

    STRATEGY: {remediation_approach} — {complexity_instruction}
    ACTION PLAN FOR THIS ANSWER:
    1) Start by addressing every item called out in the remediation report and the missed/wrong fact lists.
    2) Explicitly reference the last round ("From the previous conversation...") to note what was missed/wrong and what was recalled.
    3) Re-explain using {preferred_answer_style.replace('_', ' ')} style (different from original) and keep language simpler with concrete examples.
    4) Add 1–2 memory hooks/mnemonics.
    5) Keep to {user_topic_level} difficulty and stay concise before moving to the new question.
    </REMEDIATION_MODE>
    """

    return remediation_context


def build_evaluation_section(evaluation_report: Optional[str]) -> str:
    """Wrap the evaluation report in a consistent block for the tutor prompt."""
    if not evaluation_report:
        return ""

    return f"""
<EVALUATION_REPORT>
The following is the latest evaluation of the learner's self-assessed knowledge
across facts, strategies, procedures, and rationales. Use this to prioritize
misconceptions, omissions, and next steps, but do not quote it verbatim:

{evaluation_report}
</EVALUATION_REPORT>
    """


def get_ai_answer(
    question,
    user_profile,
    keyword_synonyms,
    remediation_mode=False,
    original_answer=None,
    original_answer_style=None,
    missed_facts=None,
    wrong_facts=None,
    recall_score=None,
    attempt_number=1,
    remediation_question=None,
    user_summary=None,
    remediation_report: Optional[str] = None,
    conversation_id: Optional[str] = None,
):
    """
    Generate personalized AI responses using Open AI with comprehensive user context.

    Process Flow:
    1. Analyze question type and extract keywords
    2. Determine optimal answer style using adaptive algorithm
    3. Build comprehensive user context for AI prompt
    4. Log all metadata for developers
    5. Generate response using Open AI

    Args:
        question (str): The user's input question
        user_profile (dict): Comprehensive user profile data
        keyword_synonyms (dict): Keyword mapping for context enhancement

        # Remediation parameters
        remediation_mode (bool): Whether this is a remediation attempt (default: False)
        original_answer (str): The original answer provided to the user (for remediation)
        original_answer_style (str): Style used in original answer (for remediation)
        missed_facts (List[str]): Facts from original answer that user failed to recall
        wrong_facts (List[str]): Facts user recalled incorrectly
        recall_score (float): User's recall score (0-100)
        attempt_number (int): Which remediation attempt this is (default: 1)
        remediation_question (Optional[str]): The question that the remediation is about (defaults to current question)
        user_summary (Optional[str]): The learner's recall/summary of the last answer
        remediation_report (Optional[str]): Structured remediation report from the previous turn
        conversation_id (Optional[str]): Remote conversation id to attach context automatically

    Returns:
        str: AI-generated personalized response or error message
    """
    try:
        # Analyze the user's question
        question_type = detect_question_type(question)
        difficulty_level = classify_question_difficulty(question, user_profile)
        print("Detected difficulty level:", difficulty_level)
        topic_label = detect_topic_label(question, keyword_synonyms)
        print("Detected topic label:", topic_label)
        update_knowledge_map(user_profile, question, topic_label, difficulty_level)
        print("Updated Knowledge Map:")
        user_topic_level = (
            user_profile.get("knowledge_map", {})
            .get(topic_label, {})
            .get("level", difficulty_level)
        )
        # Add escalation note if user has surpassed detected difficulty
        if user_topic_level == "intermediate" and difficulty_level == "beginner":
            escalation_note = "You've clearly progressed in this topic—let's build on that with a more advanced explanation."
        elif user_topic_level == "advanced" and difficulty_level in [
            "beginner",
            "intermediate",
        ]:
            escalation_note = "You've clearly mastered this topic—so let’s push into deeper theoretical territory with advanced applications."
        else:
            escalation_note = ""

        print(f"🧠 Adjusting response to topic-level: {user_topic_level}")
        from pprint import pprint

        pprint(user_profile.get("knowledge_map"))

        style_pool = ["informational", "real_world", "cause_and_effect", "goal_based"]
        relevant_styles = {
            "Fact": ["informational"],
            "Procedural": ["real_world"],
            "Rationale": ["cause_and_effect"],
            "Unknown": ["informational"],
        }.get(question_type, ["informational"])

        weights = user_profile.get("answer_type_weights", {s: 0.5 for s in style_pool})

        # Handle remediation mode - use different style than original
        if remediation_mode and original_answer_style:
            # Pick a different style to provide fresh perspective
            available_styles = [s for s in style_pool if s != original_answer_style]
            if available_styles:
                # Use weighted selection from non-original styles
                style_weights = {s: weights.get(s, 0.5) for s in available_styles}
                preferred_answer_style = max(style_weights.items(), key=lambda x: x[1])[
                    0
                ]
            else:
                preferred_answer_style = original_answer_style  # Fallback
            print(f"🔄 REMEDIATION MODE - Attempt #{attempt_number}")
            print(
                f"   Original style: {original_answer_style} → New style: {preferred_answer_style}"
            )
        elif question_type == "Fact":
            preferred_answer_style = (
                "informational"
                if weights.get("informational", 0.5) >= 0.4
                else determine_best_answer_style(user_profile)
            )
        elif random.random() < 0.8:
            preferred_answer_style = max(
                relevant_styles, key=lambda s: weights.get(s, 0)
            )
        else:
            other_styles = [s for s in style_pool if s not in relevant_styles]
            preferred_answer_style = random.choice(other_styles)

        print("Detected question type:", question_type)
        print("Selected answer style:", preferred_answer_style)
        detected_keywords = process_question_keywords(question, keyword_synonyms)

        # Knowledge base support (domain/branch pulled from preference meta or fallbacks)
        preference_meta = user_profile.get("preference_meta") or {}
        kb_domain = preference_meta.get("domain") or "general-learning"
        kb_branch = preference_meta.get("branch") or topic_label or "exploratory"
        support = collect_support_components(
            subject=kb_domain,
            dimension="Tutor",
            query=question,
            preference_meta={
                "domain": kb_domain,
                "branch": kb_branch,
                "preference": preference_meta.get("preference", ""),
            },
        )
        kb_highlights_block = (
            "\n".join(f"- {item}" for item in support.get("highlights", []) if item)
            or "- (none)"
        )
        kb_support_section = f"""
        <KNOWLEDGE_BASE_SUPPORT>
        Domain/Branch: {support.get('domain')}/{support.get('branch')}
        Highlights:
        {kb_highlights_block}
        </KNOWLEDGE_BASE_SUPPORT>
        """

        #    - Beginner: Avoid jargon, explain from first principles, use analogies or real-world metaphors.
        #    - Intermediate: Build on known foundations, include technical vocabulary with context.
        #    - Advanced: Dive into theoretical or nuanced aspects, include LaTeX math or proofs where relevant.

        # Build remediation prompt if in remediation mode
        remediation_prompt_section = ""
        if remediation_mode:
            remediation_prompt_section = build_remediation_prompt(
                question=remediation_question or question,
                user_profile=user_profile,
                missed_facts=missed_facts,
                wrong_facts=wrong_facts,
                recall_score=recall_score if recall_score is not None else 0.0,
                attempt_number=attempt_number,
                original_answer_style=original_answer_style or "informational",
                preferred_answer_style=preferred_answer_style,
                user_topic_level=user_topic_level,
                difficulty_level=difficulty_level,
                topic_label=topic_label,
                original_answer=original_answer,
                user_summary=user_summary,
                remediation_report=remediation_report,
            )

        evaluation_section = build_evaluation_section(
            user_profile.get("evaluation_report")
        )
        priority_focus = user_profile.get("priority_focus", "N/A")

        dimension_schema_block = """
        <DIMENSION_AND_LABEL_SCHEMA>
        The evaluation report organizes the learner's knowledge along four dimensions:
        - Facts: declarative knowledge of key concepts, terms, and relationships.
        - Strategies: general approaches or high-level plans for tackling problems.
        - Procedures: concrete step-by-step methods or algorithms for carrying out a strategy.
        - Rationales: explanations of why the strategies and procedures work (the "why" behind the math or logic).

        Labels within each dimension indicate how well the learner currently understands that aspect:
        - Know-Know: the learner stated something relevant and correct.
        - Know-Don't Know: the learner openly noted a relevant gap they need to learn.
        - False Knowledge: the learner asserted something relevant but wrong or misleading.
        - Omission: the learner skipped a required idea or did not mention it at all.
        - Irrelevant Knowledge: the learner brought in off-topic information that should be gently redirected.

        In your answer, use these dimension and label meanings to:
        - Acknowledge Know-Know items as strengths.
        - Address Know-Don't Know and Omission items by filling the gap clearly.
        - Correct False Knowledge in a supportive way.
        - Redirect Irrelevant Knowledge back to the core topic.
        </DIMENSION_AND_LABEL_SCHEMA>
        """

        learner_context_block = f"""
        <LEARNER_CONTEXT>
        - User topic-level: {user_topic_level}; detected difficulty: {difficulty_level}
        - Academic level: {user_profile.get('academic_level') or 'unspecified'}
        </LEARNER_CONTEXT>
        """

        if remediation_mode:
            context = f"""
<OBJECTIVE_AND_PERSONA>
You are a personal, adaptive tutor specializing in the learner's current topic.
A remediation report exists for the last turn—repair those gaps before expanding.
Current domain/branch focus: {support.get('domain')}/{support.get('branch')}
Always ground answers in BOTH the student's latest self-assessment evaluation report and their current question.
Never invent prior turns; only use the supplied remediation snapshot when referring to the last round.
</OBJECTIVE_AND_PERSONA>

<EVALUATION_AND_PRIORITY>
{evaluation_section or "No self-assessment evaluation report was provided for this learner."}

Priority focus to address first: {priority_focus}
</EVALUATION_AND_PRIORITY>

<LAST_TURN_REMEDIATION>
{remediation_prompt_section}
</LAST_TURN_REMEDIATION>

{kb_support_section}

{dimension_schema_block}
            
<INSTRUCTIONS>
- Step 1: Use the remediation report above to explicitly fix last round's misses/wrongs before anything else; mention the fixes briefly.
- Step 2: While correcting, keep the evaluation report and priority focus in mind so the fixes align to the top gaps.
- Step 3: Answer the current question once the fixes are covered, using one {preferred_answer_style.replace('_', ' ').title()}-style paragraph (3–6 sentences, ~{user_profile.get('target_fact_count', 7)} key points) tuned to {user_topic_level.title()}.
- Step 4: Align depth to the user's level. {escalation_note}
- Step 5: End by asking the learner to restate the key points they understood in their own words before moving on.
- Step 6: Leverage this conversation's history (including compaction summaries and recent items) to acknowledge what the learner already learned or fixed; avoid re-explaining covered points and build on that progress.
</INSTRUCTIONS>
            
{learner_context_block}

<CONSTRAINTS>
- Current Question: {question}
- Estimated Difficulty Level: {difficulty_level.title()}
</CONSTRAINTS>
"""
        else:
            context = f"""
<OBJECTIVE_AND_PERSONA>
You are a personal, adaptive tutor specializing in the learner's current topic.
Current domain/branch focus: {support.get('domain')}/{support.get('branch')}
Always ground answers in BOTH the student's latest self-assessment evaluation report and their current question.
Never invent prior turns. Your responses should be concise, supportive, and growth-oriented, helping the learner take the next best step.
</OBJECTIVE_AND_PERSONA>

<EVALUATION_AND_PRIORITY>
{evaluation_section or "No self-assessment evaluation report was provided for this learner."}

Highest-priority knowledge gap: {priority_focus}. Start with this before any secondary topics.
</EVALUATION_AND_PRIORITY>

{kb_support_section}

{dimension_schema_block}
            
<INSTRUCTIONS>  
Answer Content:
- Anchor the answer in the evaluation report and PRIORITY_FOCUS first; surface at least one strength and one gap.
- Provide one focused explanation that fills the top gap for this question.
- Priority order: redirect irrelevant ideas → correct false knowledge → fill omissions → reinforce confidence. Do not solve the entire task; end with one high-leverage next step tied to their words.
- Leverage this conversation's history (including compaction summaries and recent items): acknowledge what the learner already learned or corrected; avoid repeating covered points; build on the latest progress.

Answer Style:
- Output one {preferred_answer_style.replace('_', ' ').title()}-style paragraph, 3–6 sentences, ~{user_profile.get('target_fact_count', 7)} key points, tuned to {user_topic_level.title()}.
- Align depth to the user's level. {escalation_note}
- Adapt examples/metaphors to the learner's learning style and language {user_profile.get('language')}.
- End by asking the learner to restate the key points they understood in their own words before moving on.
</INSTRUCTIONS>
            
{learner_context_block}

<CONSTRAINTS>
- Current Question: {question}
- Estimated Difficulty Level: {difficulty_level.title()}
</CONSTRAINTS>
"""

        # Debug: print full prompt to terminal so we can inspect it
        try:
            print("\n" + "=" * 60)
            print(
                f"[Q&A Prompt] User: {user_profile.get('username')} | "
                f"Domain/Branch: {support.get('domain')}/{support.get('branch')} | "
                f"Question: {question}"
            )
            print("-" * 60)
            print(context)
            print("=" * 60 + "\n")
            logging.info(
                "[Q&A Prompt] domain=%s branch=%s user=%s question='%.120s'",
                support.get("domain"),
                support.get("branch"),
                user_profile.get("username"),
                question,
            )
        except Exception:
            # Avoid crashing if printing fails for any reason
            pass

        # Generate AI response using Chat Completions API (works with both OpenAI and Ollama)
        try:
            response = get_client().chat.completions.create(
                model=get_model(),
                messages=[{"role": "user", "content": context}],
                temperature=0.7,
                max_tokens=1000,
            )
            ai_output = (response.choices[0].message.content or "").strip()
            if not ai_output:
                ai_output = "Sorry, no answer was generated."
        except Exception as exc:
            logging.error(f"LLM Chat Completions Error: {exc}")
            return "Sorry, there was an issue generating the answer. Please try again later."
        # Print LLM reply to terminal/VS Code for inspection during normal Q&A
        try:
            print("\n" + "=" * 100)
            print(
                f"[Q&A Reply] User: {user_profile.get('username')} | Question: {question}"
            )
            print("-" * 100)
            print(ai_output)
            print("=" * 100 + "\n")
        except Exception:
            pass
        return ai_output

    except Exception as e:
        logging.error(f"Open AI API Error: {str(e)}")
        return (
            "Sorry, there was an issue generating the answer. Please try again later."
        )


def update_knowledge_map(user_profile, question, topic_label, difficulty_level):
    """Update knowledge map."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if "knowledge_map" not in user_profile or user_profile["knowledge_map"] is None:
        user_profile["knowledge_map"] = {}

    topic_entry = user_profile["knowledge_map"].get(
        topic_label, {"level": difficulty_level, "last_asked": timestamp, "history": []}
    )

    topic_entry["last_asked"] = timestamp
    topic_entry["history"].append(
        {"difficulty": difficulty_level, "timestamp": timestamp, "question": question}
    )

    print(f"📈 History length for '{topic_label}': {len(topic_entry['history'])}")

    # 🔁 Escalate difficulty based on recent beginner history
    recent_entries = topic_entry["history"][-3:]
    beginner_count = sum(1 for h in recent_entries if h["difficulty"] == "beginner")

    # if topic_entry["level"] == "beginner" and beginner_count >= 3:
    #     print("🔼 Escalating difficulty to INTERMEDIATE!")
    #     topic_entry["level"] = "intermediate"
    # elif topic_entry["level"] == "intermediate":
    #     intermediate_count = sum(1 for h in recent_entries if h["difficulty"] == "intermediate")
    #     if intermediate_count >= 3:
    #         print("🚀 Escalating difficulty to ADVANCED!")
    #         topic_entry["level"] = "advanced"

    current_level = topic_entry.get("level", "beginner")

    if current_level == "beginner" and len(topic_entry["history"]) >= 3:
        topic_entry["level"] = "intermediate"
        print("🔼 Escalating difficulty to INTERMEDIATE!")

    elif current_level == "intermediate" and len(topic_entry["history"]) >= 6:
        topic_entry["level"] = "advanced"
        print("🔼 Escalating difficulty to ADVANCED!")

    user_profile["knowledge_map"][topic_label] = topic_entry
