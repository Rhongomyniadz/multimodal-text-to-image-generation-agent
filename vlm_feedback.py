import base64
import json
from google import genai
from google.genai import types
from pydantic import BaseModel

import pipeline

class FeedbackResult(BaseModel):
    passed: bool
    reason: str
    missing_elements: str

def analyze_image(image_b64: str, user_request: str):
    """
    Uses Gemini VLM to compare the generated image against the user's request.
    """
    if not pipeline.config['visual_feedback']['enabled']:
        return FeedbackResult(passed=True, reason="Feedback loop disabled", missing_elements="")

    image_bytes = base64.b64decode(image_b64)

    SYSTEM_INSTRUCTION = """
    You are an AI Art Quality Assurance Auditor.
    
    Task: 
    Compare the provided IMAGE against the USER REQUEST.
    Identify if any KEY subjects or specific details mentioned in the text are completely missing from the image.
    
    Strictness:
    - Ignore minor style differences.
    - Focus on MISSING OBJECTS (e.g., User asked for "a cat in a hat", Image has no hat).
    - Focus on WRONG COLORS (e.g., User asked for "Red car", Image is Blue).

    Output Schema (JSON):
    {
        "passed": boolean, (true if acceptable, false if major errors)
        "reason": "Short explanation",
        "missing_elements": "What specifically needs to be added/fixed?"
    }
    """

    prompt = f"User Request: '{user_request}'\nDoes the image satisfy this request?"

    try:
        response = pipeline.google_client.models.generate_content(
            model=pipeline.config['visual_feedback']['vlm_model'],
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        types.Part(text=prompt)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=FeedbackResult,
                temperature=0.1,
            )
        )
        
        return response.parsed

    except Exception as e:
        print(f"[ERROR] VLM Analysis Failed: {e}")
        return FeedbackResult(passed=True, reason=f"Analysis failed: {e}", missing_elements="")