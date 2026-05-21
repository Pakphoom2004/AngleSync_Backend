import os
import mimetypes
import json

from dotenv import load_dotenv

try:
    from google import genai as google_genai
except ImportError:
    google_genai = None

legacy_genai = None
if google_genai is None:
    try:
        import google.generativeai as legacy_genai
    except ImportError:
        legacy_genai = None

from exceptions.service_exception import (
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

Analysis Data:
{analysis_data}
""".strip()


def _build_json_feedback_prompt(
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


def _get_api_key():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ServiceException(
            "Missing GEMINI_API_KEY in environment."
        )

    return api_key


def _extract_response_text(
        response
):

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


def _empty_feedback():

    return {
        key: ""
        for key in FEEDBACK_KEYS
    }


def _normalize_feedback(
        feedback
):

    normalized = _empty_feedback()

    if not isinstance(
            feedback,
            dict
    ):
        return normalized

    for key in FEEDBACK_KEYS:
        value = feedback.get(
            key,
            ""
        )
        if value is None:
            value = ""
        normalized[key] = str(
            value
        ).strip()

    return normalized


def _is_feedback_complete(
        feedback: dict
):

    return all(
        feedback.get(key, "").strip()
        for key in FEEDBACK_KEYS
    )


def _parse_json_feedback(
        text: str
):

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    feedback = _normalize_feedback(
        parsed
    )

    if _is_feedback_complete(
            feedback
    ):
        return feedback

    return None


def _joint_issue_labels(
        angles: dict
):

    issue_labels = []

    if not angles:
        return issue_labels

    if (
        angles.get("left_knee", 90) > 150
        or angles.get("right_knee", 90) > 150
    ):
        issue_labels.append(
            "limited knee flexion"
        )

    if (
        angles.get("left_shoulder", 40) > 80
        or angles.get("right_shoulder", 40) > 80
    ):
        issue_labels.append(
            "shoulder misalignment"
        )

    if (
        angles.get("left_hip", 100) > 150
        or angles.get("right_hip", 100) > 150
    ):
        issue_labels.append(
            "reduced hip loading"
        )

    if (
        angles.get("left_elbow", 160) < 45
        or angles.get("right_elbow", 160) < 45
    ):
        issue_labels.append(
            "arm instability"
        )

    return issue_labels


def build_rule_based_feedback(
        analysis_data: dict
):

    risk_score = float(
        analysis_data.get(
            "risk_score",
            0
        )
    )
    angles = analysis_data.get(
        "angles",
        {}
    )

    issue_labels = _joint_issue_labels(
        angles
    )
    issue_text = ", ".join(
        issue_labels[:3]
    ) or "general movement instability"

    if risk_score >= 80:
        severity = "high"
        form_summary = (
            "Form control is poor at the highest-risk frame with clear signs of "
            f"{issue_text}."
        )
        injury_risk = (
            "Injury risk is high because the position shows excessive deviation from the target angles."
        )
        corrective_cues = (
            "Slow the rep, keep the knees tracking forward, brace the trunk, and stay balanced through the feet."
        )
        practice_plan = (
            "Reduce depth or load, rehearse 2-3 slow controlled reps per set, and rebuild range before adding intensity."
        )
    elif risk_score >= 60:
        severity = "moderate"
        form_summary = (
            "Form is inconsistent at the highest-risk frame with noticeable "
            f"{issue_text}."
        )
        injury_risk = (
            "Injury risk is moderate if the same alignment errors repeat under fatigue."
        )
        corrective_cues = (
            "Control the tempo, keep the shoulders and hips aligned, and avoid locking the joints too early."
        )
        practice_plan = (
            "Use lighter practice sets and pause briefly in the bottom position to improve control."
        )
    else:
        severity = "low"
        form_summary = (
            "Form is mostly stable with only minor signs of "
            f"{issue_text}."
        )
        injury_risk = (
            "Injury risk is currently low, but consistency should be maintained through the full movement."
        )
        corrective_cues = (
            "Keep the same posture pattern, stay braced, and maintain smooth joint tracking."
        )
        practice_plan = (
            "Continue with steady technique work and add load only if the same control is maintained."
        )

    feedback = _normalize_feedback({
        "form_summary": form_summary,
        "injury_risk": injury_risk,
        "corrective_cues": corrective_cues,
        "practice_plan": practice_plan
    })

    return {
        "feedback": feedback,
        "feedback_source": "rule_based",
        "feedback_severity": severity
    }


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

    client = google_genai.Client(
        api_key=_get_api_key()
    )

    contents = [prompt]

    if os.path.exists(
            frame_path
    ):

        inline_image = _build_inline_image_part(
            frame_path
        )

        contents.append(
            inline_image
        )

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=google_genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    return _extract_response_text(
        response
    )


def _generate_with_legacy_genai(
        prompt: str,
        frame_path: str,
        model_name: str
):

    legacy_genai.configure(
        api_key=_get_api_key()
    )

    model = legacy_genai.GenerativeModel(
        model_name
    )

    contents = [prompt]

    if os.path.exists(
            frame_path
    ):

        uploaded_file = legacy_genai.upload_file(
            path=frame_path
        )

        contents.append(
            uploaded_file
        )

    response = model.generate_content(
        contents
    )

    return _extract_response_text(
        response
    )


def parse_feedback_sections(
        text: str
):

    sections = _empty_feedback()

    current_section = None

    for line in text.splitlines():

        line = line.strip()

        if line.startswith(
                "Form Summary:"
        ):

            current_section = (
                "form_summary"
            )

            sections[
                current_section
            ] = line.replace(
                "Form Summary:",
                ""
            ).strip()

        elif line.startswith(
                "Injury Risk:"
        ):

            current_section = (
                "injury_risk"
            )

            sections[
                current_section
            ] = line.replace(
                "Injury Risk:",
                ""
            ).strip()

        elif line.startswith(
                "Corrective Cues:"
        ):

            current_section = (
                "corrective_cues"
            )

            sections[
                current_section
            ] = line.replace(
                "Corrective Cues:",
                ""
            ).strip()

        elif line.startswith(
                "Practice Plan:"
        ):

            current_section = (
                "practice_plan"
            )

            sections[
                current_section
            ] = line.replace(
                "Practice Plan:",
                ""
            ).strip()

        else:

            if current_section:

                sections[
                    current_section
                ] += (
                    " " + line
                )

    return _normalize_feedback(
        sections
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
    json_prompt = _build_json_feedback_prompt(
        analysis_data,
        prompt_template
    )

    try:

        if google_genai is not None:
            response_text = (
                _generate_with_google_genai(
                    json_prompt,
                    frame_path,
                    model_name
                )
            )
        elif legacy_genai is not None:
            response_text = (
                _generate_with_legacy_genai(
                    prompt,
                    frame_path,
                    model_name
                )
            )
        else:
            raise ServiceException(
                "No Gemini SDK installed. Install google-genai or google-generativeai."
            )

        feedback = (
            _parse_json_feedback(
                response_text
            )
            or parse_feedback_sections(
                response_text
            )
        )

        if _is_feedback_complete(
                feedback
        ):
            return {
                "prompt": prompt,
                "feedback": feedback,
                "feedback_source": "gemini"
            }

    except Exception as error:

        print("Gemini API Error:")
        print(error)

        fallback = build_rule_based_feedback(
            analysis_data
        )
        fallback["prompt"] = prompt
        fallback["feedback_error"] = str(error)
        return fallback

    fallback = build_rule_based_feedback(
        analysis_data
    )
    fallback["prompt"] = prompt
    fallback["feedback_error"] = (
        "Gemini returned incomplete feedback."
    )
    return fallback
