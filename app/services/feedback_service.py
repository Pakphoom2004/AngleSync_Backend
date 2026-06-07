import json
import mimetypes
import os

from dotenv import load_dotenv
from google import genai as google_genai

from app.exceptions import (
    ServiceException
)

load_dotenv()

PROMPT_TEMPLATE = """
You are an expert fitness coach.

Analyze posture using:
- joint angles
- posture alignment
- movement stability
- injury risk

Focus on:
- shoulders
- hips
- knees
- spine

Return EXACTLY:

Form Summary:
Injury Risk:
Corrective Cues:
Practice Plan:

Keep responses short and professional.
"""

FEEDBACK_KEYS = (
    "form_summary",
    "injury_risk",
    "corrective_cues",
    "practice_plan"
)

DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.0-flash"
)


def build_feedback_prompt(
        analysis_data: dict,
        prompt_template: str = PROMPT_TEMPLATE
):

    return f"""
{prompt_template}

Return JSON only with these exact keys:
- form_summary
- injury_risk
- corrective_cues
- practice_plan

Rules:
- Output valid JSON only.
- Each value must be a short string.
- Do not use markdown.

Analysis Data:
{analysis_data}
""".strip()


def _build_inline_image_part(
        frame_path: str
):

    mime_type = (
        mimetypes.guess_type(frame_path)[0]
        or "application/octet-stream"
    )

    with open(
        frame_path,
        "rb"
    ) as image_file:
        image_bytes = image_file.read()

    return google_genai.types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

def _generate_with_google_genai(
        prompt: str,
        frame_path: str,
        model_name: str
):

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ServiceException(
            "Missing GEMINI_API_KEY in environment."
        )

    client = google_genai.Client(
        api_key=api_key
    )

    contents = [prompt]

    if os.path.exists(
            frame_path
    ):
        contents.append(
            _build_inline_image_part(
                frame_path
            )
        )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=google_genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
    except Exception as error:
        raise ServiceException(
            str(error)
        ) from error

    response_text = getattr(
        response,
        "text",
        None
    )

    if response_text:
        return response_text.strip()

    raise ServiceException(
        "Gemini returned an empty response."
    )


def parse_feedback_response(
        text: str
):

    feedback = {
        key: ""
        for key in FEEDBACK_KEYS
    }

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(
            parsed,
            dict
    ):
        for key in FEEDBACK_KEYS:
            value = parsed.get(
                key,
                ""
            )
            feedback[key] = str(
                value or ""
            ).strip()
    else:
        current_section = None

        for line in text.splitlines():

            line = line.strip()

            if line.startswith(
                    "Form Summary:"
            ):
                current_section = (
                    "form_summary"
                )
                feedback[current_section] = line.replace(
                    "Form Summary:",
                    ""
                ).strip()
            elif line.startswith(
                    "Injury Risk:"
            ):
                current_section = (
                    "injury_risk"
                )
                feedback[current_section] = line.replace(
                    "Injury Risk:",
                    ""
                ).strip()
            elif line.startswith(
                    "Corrective Cues:"
            ):
                current_section = (
                    "corrective_cues"
                )
                feedback[current_section] = line.replace(
                    "Corrective Cues:",
                    ""
                ).strip()
            elif line.startswith(
                    "Practice Plan:"
            ):
                current_section = (
                    "practice_plan"
                )
                feedback[current_section] = line.replace(
                    "Practice Plan:",
                    ""
                ).strip()
            elif current_section:
                feedback[current_section] += (
                    " " + line
                )

    is_complete = all(
        feedback.get(key, "").strip()
        for key in FEEDBACK_KEYS
    )

    return (
        feedback,
        is_complete
    )


def generate_advanced_feedback(
        analysis_data: dict,
        frame_path: str,
        prompt_template: str = PROMPT_TEMPLATE,
        model_name: str = DEFAULT_MODEL
):
    prompt = build_feedback_prompt(
        analysis_data,
        prompt_template
    )

    try:
        response_text = (
            _generate_with_google_genai(
                prompt,
                frame_path,
                model_name
            )
        )

        feedback, is_complete = (
            parse_feedback_response(
                response_text
            )
        )

        if is_complete:
            return {
                "prompt": prompt,
                "feedback": feedback
            }

    except ServiceException:
        raise
