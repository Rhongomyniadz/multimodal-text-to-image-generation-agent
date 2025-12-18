from __future__ import annotations

from typing import Any, Dict

from google.genai import types

from langchain_core.runnables import RunnableLambda

import pipeline
import vlm_feedback


DEFAULT_SYSTEM_INSTRUCTION = """
You are an expert SDXL Prompt Engineer with persistent memory.
ALWAYS check history. Output structured JSON {positive, negative}.
""".strip()


def create_brain_chat(history, system_instruction: str) -> Any:
    return pipeline.google_client.chats.create(
        model=pipeline.config["models"]["brain"],
        history=history,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=pipeline.SDXLPrompt,
            temperature=0.7,
        ),
    )


def brain_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Brain step: create chat (with history) and generate an optimized SDXLPrompt."""
    user_input = state["user_input"]
    history = state.get("history", [])
    system_instruction = state.get("system_instruction", DEFAULT_SYSTEM_INSTRUCTION)

    pipeline.logger.info("[BRAIN] Analyzing context and optimizing prompt...")
    chat = create_brain_chat(history=history, system_instruction=system_instruction)

    response = chat.send_message(user_input)
    prompt_data = response.parsed

    pipeline.logger.info("[BRAIN] Optimization Complete.")
    pipeline.logger.info(f"   -> Original: '{user_input}'")
    pipeline.logger.info(f"   -> Optimized: '{prompt_data.positive}...'")

    # Keep chat for potential correction round.
    state.update(
        {
            "_chat": chat,
            "prompt_data": prompt_data,
            "brain_response_text": response.text,
        }
    )
    return state


def paint_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Painter step: call SDXL generation API using the optimized prompt."""
    pipeline.logger.info("[PAINTER] Painting (Attempt 1)...")
    image_b64, error_msg = pipeline.generate_image_sdxl(state["prompt_data"])
    if error_msg:
        raise RuntimeError(error_msg)
    state["image_b64"] = image_b64
    return state


def critic_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Critic step: run VLM inspection against the user request."""
    pipeline.logger.info("[CRITIC] Starting Visual Inspection...")
    critique = vlm_feedback.analyze_image(state["image_b64"], state["user_input"])
    state["critique"] = critique
    return state


def auto_fix_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-fix step: if Critic fails, regenerate prompt and repaint once."""
    critique = state.get("critique")
    if critique is None or critique.passed:
        pipeline.logger.info("[CRITIC] PASS: Image looks correct.")
        state["auto_fixed"] = False
        return state

    pipeline.logger.warning(
        f"[CRITIC] FAIL: {critique.reason}. Missing: {critique.missing_elements}"
    )

    chat = state.get("_chat")
    if chat is None:
        # Should never happen
        raise RuntimeError("Internal error: chat object missing for auto-fix.")

    correction_prompt = f"""
SYSTEM ALERT: The previous image failed visual inspection.
Reason: {critique.reason}.
Missing Elements: {critique.missing_elements}.
TASK: Regenerate the SDXL JSON prompt.
Keep the style, but STRONG EMPHASIS on including: {critique.missing_elements}.
""".strip()

    pipeline.logger.info("[BRAIN] Applying fix and regenerating prompt...")
    response_fix = chat.send_message(correction_prompt)
    prompt_data_fix = response_fix.parsed

    pipeline.logger.info("[PAINTER] Painting (Attempt 2 - Fixed)...")
    image_b64_fix, error_msg_fix = pipeline.generate_image_sdxl(prompt_data_fix)

    if error_msg_fix:
        pipeline.logger.error("[SYSTEM] Auto-fix generation failed; using Attempt 1 result.")
        state["auto_fixed"] = False
        state["auto_fix_error"] = error_msg_fix
        return state

    pipeline.logger.info("[SYSTEM] Auto-fix successful.")
    state.update(
        {
            "auto_fixed": True,
            "prompt_data": prompt_data_fix,
            "image_b64": image_b64_fix,
            "brain_response_text": response_fix.text,
        }
    )
    return state


def build_runnables():
    """Return LangChain runnables for each stage (so UI can show step-by-step status)."""
    return {
        "brain": RunnableLambda(brain_step),
        "paint": RunnableLambda(paint_step),
        "critic": RunnableLambda(critic_step),
        "auto_fix": RunnableLambda(auto_fix_step),
    }


def build_full_chain():
    """Return a single runnable chain that executes Brain -> Painter -> Critic -> Auto-fix."""
    r = build_runnables()

    def maybe_critic(state: Dict[str, Any]) -> Dict[str, Any]:
        if not state.get("enable_feedback", True):
            # Mimic a pass result for consistent downstream behavior.
            state["critique"] = vlm_feedback.FeedbackResult(
                passed=True,
                reason="Feedback loop disabled",
                missing_elements="",
            )
            return state
        return r["critic"].invoke(state)

    return r["brain"] | r["paint"] | RunnableLambda(maybe_critic) | r["auto_fix"]
