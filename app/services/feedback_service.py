import base64
import json
import mimetypes
import os
import time


from dotenv import load_dotenv
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from app.exceptions import ServiceException

load_dotenv()

PROMPT_TEMPLATE = """
You are an expert fitness coach analyzing posture from movement data.

Analyze the following using joint angles, alignment, stability, and injury risk.
Focus on: shoulders, hips, knees, and spine.

Respond with valid JSON only. No markdown, no extra text.

Required keys:
- "form_summary": overall posture assessment
- "injury_risk": identified risk areas
- "corrective_cues": actionable corrections
- "practice_plan": recommended next steps

Keep each value concise and professional.
"""

FEEDBACK_KEYS = (
    "form_summary",
    "injury_risk",
    "corrective_cues",
    "practice_plan"
)

FEEDBACK_WRAPPER_KEYS = (
    "feedback",
    "analysis",
    "result",
    "data"
)

MARKDOWN_HEADING_PREFIXES = (
    "#",
    "*",
    "-"
)

DEFAULT_MODEL = os.getenv(
    "ZAI_MODEL",
)

def build_feedback_prompt(analysis_data, prompt_template=PROMPT_TEMPLATE):
    return f"{prompt_template}\n\nAnalysis Data:\n{analysis_data}".strip()


def _clean_markdown_line(line):
    cleaned = line.strip()
    cleaned = cleaned.lstrip("#").strip()
    cleaned = cleaned.lstrip("*").strip()
    cleaned = cleaned.lstrip("-").strip()

    if ".  " in cleaned[:5]:
        cleaned = cleaned.split(".  ", 1)[1].strip()

    return cleaned.replace("**", "").strip()


def _format_markdown_lines(lines):
    return " ".join(
        _clean_markdown_line(line)
        for line in lines
        if _clean_markdown_line(line)
    ).strip()


def _extract_markdown_section(lines, start_markers, end_markers=()):
    selected = []
    in_section = False

    for line in lines:
        normalized = _clean_markdown_line(line).lower()

        if not in_section and any(marker in normalized for marker in start_markers):
            in_section = True
            selected.append(line)
            continue

        if in_section and end_markers and any(
            marker in normalized for marker in end_markers
        ):
            break

        if in_section:
            selected.append(line)

    return _format_markdown_lines(selected)


def _parse_markdown_feedback(cleaned):
    lines = [
        line
        for line in cleaned.splitlines()
        if _clean_markdown_line(line)
    ]

    if not lines:
        return None

    risk_section = _extract_markdown_section(
        lines,
        ("overall risk assessment", "risk score", "injury risk"),
        ("detailed posture analysis", "recommended correction")
    )
    detail_section = _extract_markdown_section(
        lines,
        ("detailed posture analysis",),
        ("recommended correction",)
    )
    correction_section = _extract_markdown_section(
        lines,
        ("recommended correction", "recommended corrections", "correction")
    )

    readable_text = _format_markdown_lines(lines)

    if not any((risk_section, detail_section, correction_section)):
        return None

    form_summary = detail_section or readable_text
    injury_risk = risk_section or form_summary
    corrective_cues = correction_section or form_summary
    practice_plan = correction_section or corrective_cues

    return {
        "form_summary": form_summary,
        "injury_risk": injury_risk,
        "corrective_cues": corrective_cues,
        "practice_plan": practice_plan
    }


def _generate_with_ai(prompt, model_name):
    if OpenAI is None:
        raise ServiceException("Missing openai package dependency.")

    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        raise ServiceException("Missing ZAI_API_KEY in environment.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.z.ai/api/coding/paas/v4"  # Coding Plan endpoint
    )

    content = [{"type": "text", "text": prompt}]

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0.2
            )

            response_text = response.choices[0].message.content
            if response_text:
                return response_text.strip()

        except Exception as error:
            is_rate_limited = (
                getattr(error, "status_code", None) == 429
                or "429" in str(error)
            )
            if is_rate_limited:
                if attempt < 2:
                    wait = 5 * (attempt + 1)  # 5s, 10s
                    print(f"[DEBUG] Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise ServiceException(
                    "Z.AI rate limit exceeded after 3 attempts. "
                    "Please try again later or check the API quota."
                ) from error
            raise ServiceException(str(error)) from error

    raise ServiceException("Z.AI rate limit exceeded after retries.")

def parse_feedback_response(
        text: str
):
    feedback = {
        key: ""
        for key in FEEDBACK_KEYS
    }

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        for wrapper_key in FEEDBACK_WRAPPER_KEYS:
            nested = parsed.get(wrapper_key)
            if isinstance(nested, dict):
                parsed = nested
                break

    if isinstance(parsed, dict):
        for key in FEEDBACK_KEYS:
            value = parsed.get(key, "")
            feedback[key] = str(value or "").strip()
    else:
        current_section = None

        for line in cleaned.splitlines():
            line = line.strip()

            if line.startswith("Form Summary:"):
                current_section = "form_summary"
                feedback[current_section] = line.replace(
                    "Form Summary:", ""
                ).strip()
            elif line.startswith("Injury Risk:"):
                current_section = "injury_risk"
                feedback[current_section] = line.replace(
                    "Injury Risk:", ""
                ).strip()
            elif line.startswith("Corrective Cues:"):
                current_section = "corrective_cues"
                feedback[current_section] = line.replace(
                    "Corrective Cues:", ""
                ).strip()
            elif line.startswith("Practice Plan:"):
                current_section = "practice_plan"
                feedback[current_section] = line.replace(
                    "Practice Plan:", ""
                ).strip()
            elif current_section:
                feedback[current_section] += " " + line

        if not all(feedback.get(key, "").strip() for key in FEEDBACK_KEYS):
            markdown_feedback = _parse_markdown_feedback(cleaned)
            if markdown_feedback is not None:
                feedback.update(markdown_feedback)

    is_complete = all(
        feedback.get(key, "").strip()
        for key in FEEDBACK_KEYS
    )

    return feedback, is_complete

def generate_advanced_feedback(
        analysis_data: dict,
        prompt_template: str = PROMPT_TEMPLATE,
        model_name: str = DEFAULT_MODEL
):
    prompt = build_feedback_prompt(
        analysis_data,
        prompt_template
    )

    try:
        response_text = _generate_with_ai(
            prompt,
            model_name
        )
        print(f"[DEBUG] response_text: {response_text}")

        feedback, is_complete = parse_feedback_response(
            response_text
        )
        print(f"[DEBUG] is_complete: {is_complete}")       
        print(f"[DEBUG] feedback: {feedback}") 
        if not is_complete:
            raise ServiceException("Incomplete feedback from model.")

        if is_complete:
            return {
                "prompt": prompt,
                "feedback": feedback
            }

    except ServiceException:
        raise
    except Exception as e:                                
        print(f"[DEBUG] error in generate_advanced_feedback: {type(e).__name__}: {e}")
        raise ServiceException(str(e))
