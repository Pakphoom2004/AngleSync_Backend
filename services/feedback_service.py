import google.generativeai as genai

from exceptions.service_exception import ServiceException

# Gemini prompt template
PROMPT_TEMPLATE = """
You are an expert sports biomechanics specialist, certified strength coach, and physical therapist.

Analyze the user's exercise posture and movement quality using:
- body joint angles
- movement alignment
- joint stability
- posture symmetry
- movement compensation
- frame-by-frame risk analysis
- highest-risk frame image

Focus your analysis on:
- shoulder alignment
- hip alignment
- knee tracking
- spine posture
- joint coordination
- movement balance
- injury prevention

Evaluate whether the movement demonstrates:
- improper posture
- unstable joints
- limited mobility
- muscular imbalance
- unsafe exercise mechanics

Provide professional and specific coaching feedback.

Return the result in EXACTLY these 4 sections:

Form Summary:
Summarize the user's overall movement quality, posture stability, and exercise execution.

Injury Risk:
Explain possible injury risks caused by incorrect movement mechanics, posture deviation, instability, or joint stress.

Corrective Cues:
Provide clear and actionable posture correction cues the user can immediately apply during exercise.

Practice Plan:
Recommend a short training or mobility plan to improve posture, stability, movement control, and exercise safety.

Rules:
- Do not add extra sections.
- Do not use markdown.
- Keep each section concise but professional.
- Use simple fitness coaching language understandable to normal users.
"""

# Configure Gemini API
genai.configure(
    api_key="YOUR_GEMINI_API_KEY"
)

model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

# Parse Gemini response into sections
def parse_feedback_sections(text: str):

    sections = {
        "form_summary": "",
        "injury_risk": "",
        "corrective_cues": "",
        "practice_plan": ""
    }

    current_section = None

    for line in text.splitlines():

        line = line.strip()

        if line.startswith("Form Summary:"):
            current_section = "form_summary"
            sections[current_section] = line.replace(
                "Form Summary:",
                ""
            ).strip()

        elif line.startswith("Injury Risk:"):
            current_section = "injury_risk"
            sections[current_section] = line.replace(
                "Injury Risk:",
                ""
            ).strip()

        elif line.startswith("Corrective Cues:"):
            current_section = "corrective_cues"
            sections[current_section] = line.replace(
                "Corrective Cues:",
                ""
            ).strip()

        elif line.startswith("Practice Plan:"):
            current_section = "practice_plan"
            sections[current_section] = line.replace(
                "Practice Plan:",
                ""
            ).strip()

        else:
            if current_section:
                sections[current_section] += (
                    " " + line
                )

    return sections

# Generate advanced AI feedback
def generate_advanced_feedback(
        analysis_data: dict,
        frame_path: str,
        prompt_template: str = PROMPT_TEMPLATE
):

    try:

        uploaded_file = genai.upload_file(
            path=frame_path
        )

        prompt = f"""
        {prompt_template}

        Analysis Data:
        {analysis_data}
        """

        response = model.generate_content([
            uploaded_file,
            prompt
        ])

        response_text = response.text.strip()

        feedback = parse_feedback_sections(
            response_text
        )

        return feedback

    except Exception:
        raise ServiceException()