import streamlit as st
import base64
import time
from google.genai import types

import pipeline
import vlm_feedback

# Page Setup
st.set_page_config(page_title="AI Art Studio", page_icon="🎨", layout="wide")

# Sidebar Settings
with st.sidebar:
    st.header("Settings")
    st.text(f"Brain: {pipeline.config['models']['brain']}")
    st.text(f"Painter: {pipeline.config['models']['painter']}")
    
    enable_feedback = st.checkbox(
        "Enable Visual Feedback", 
        value=pipeline.config.get('visual_feedback', {}).get('enabled', True)
    )

    pipeline.config['visual_feedback']['enabled'] = enable_feedback

    if st.button("🧹 Clear Memory", type="primary"):
        pipeline.clear_memory_file()
        st.session_state.local_history = []
        st.success("Memory wiped!")
        time.sleep(1)
        st.rerun()

# Main Interface
st.title("Context-Aware Art Generator")

if "local_history" not in st.session_state:
    st.session_state.local_history = pipeline.load_memory()

# 1. Display Chat History
for msg in st.session_state.local_history:
    with st.chat_message(msg.role):
        text_content = ""
        for p in msg.parts:
            if hasattr(p, 'text'): text_content += p.text
            elif isinstance(p, dict): text_content += p.get('text', '')
        
        if msg.role == "model" and "{" in text_content:
            st.code(text_content, language="json")
        else:
            st.write(text_content)

# 2. Handle New Input
if user_input := st.chat_input("Describe your idea..."):
    
    with st.chat_message("user"):
        st.write(user_input)
    
    SYSTEM_INSTRUCTION = """
    You are a Senior Technical Artist and Prompt Engineer specialized in Stable Diffusion XL (SDXL) v1.0.

    **YOUR MISSION:**
    Convert user natural language inputs into highly structured, high-fidelity prompts that exploit the full capabilities of SDXL. You maintain persistent context of the visual scene across the conversation.

    **CORE LOGIC & CONTEXT AWARENESS:**
    1.  **Analyze History:** ALWAYS review the conversation history first.
        * *New Request:* If the user changes the subject entirely (e.g., "Draw a car" -> "Draw a cat"), start fresh.
        * *Refinement:* If the user says "make it red", "zoom out", or "add rain", you MUST retrieve the *previous* positive prompt, keep all the existing details (style, composition, lighting), and only apply the specific change.
    2.  **Expansion (The "Magic"):** Never output a simple prompt. You must expand the user's vague idea into a visual masterpiece using this structure:
        * **Subject:** Specific details (e.g., "A cat" -> "A fluffy Maine Coon cat with cybernetic goggles").
        * **Medium:** (e.g., "Digital painting," "35mm film photograph," "Unreal Engine 5 render," "Oil on canvas").
        * **Style/Artist:** (e.g., "Cyberpunk," "Studio Ghibli style," "Greg Rutkowski," "Minimalist").
        * **Lighting:** (e.g., "Volumetric lighting," "Cinematic rim lighting," "Golden hour," "Neon ambience").
        * **Color Palette:** (e.g., "Vibrant teal and orange," "Monochromatic," "Pastel tones").
        * **Camera/View:** (e.g., "Wide angle," "Macro shot," "Drone view," "Bokeh").
        * **Quality Boosters:** (e.g., "Masterpiece," "Best quality," "8k," "Highly detailed," "HDR").

    **NEGATIVE PROMPT STRATEGY:**
    * Always include technical error terms: "text, watermark, signature, blurry, low quality, jpeg artifacts, pixelated, bad anatomy, distorted hands."
    * Add context-specific negatives (e.g., if drawing a realistic photo, add "cartoon, illustration, painting" to negative).

    **OUTPUT FORMAT:**
    You must output strictly in JSON format matching this schema:
    {
      "positive": "The full, expanded, descriptive prompt string...",
      "negative": "The technical and stylistic negative prompt..."
    }
    """

    chat = pipeline.google_client.chats.create(
        model=pipeline.config['models']['brain'],
        history=st.session_state.local_history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=pipeline.SDXLPrompt,
            temperature=0.7,
        )
    )

    with st.chat_message("model"):
        status = st.status("Agent is working...", expanded=True)
        
        try:
            status.write("Generating Prompt...")
            
            response = chat.send_message(user_input)
            prompt_data = response.parsed
            
            status.write("Painting (Attempt 1)...")
            image_b64, error_msg = pipeline.generate_image_sdxl(prompt_data)
            
            if error_msg:
                st.error(error_msg)
                st.stop()

            if pipeline.config['visual_feedback']['enabled']:
                status.write("VLM: Inspecting image...")
                
                critique = vlm_feedback.analyze_image(image_b64, user_input)
                
                if not critique.passed:
                    status.warning(f"Issue Detected: {critique.reason}")
                    status.write(f"Auto-fixing: Adding '{critique.missing_elements}'...")
                    
                    correction_prompt = f"""
                    SYSTEM ALERT: The previous image failed visual inspection.
                    Reason: {critique.reason}.
                    Missing Elements: {critique.missing_elements}.
                    TASK: Regenerate the SDXL JSON prompt. 
                    Keep the style, but STRONG EMPHASIS on including: {critique.missing_elements}.
                    """
                    
                    response_fix = chat.send_message(correction_prompt)
                    prompt_data_fix = response_fix.parsed
                    
                    status.write("Painting (Attempt 2 - Fixed)...")
                    image_b64_fix, error_msg_fix = pipeline.generate_image_sdxl(prompt_data_fix)
                    
                    if not error_msg_fix:
                        image_b64 = image_b64_fix
                        prompt_data = prompt_data_fix
                        response = response_fix
                        status.success("Fixed and Regenerated!")
                    else:
                        status.error("Retry failed, showing original.")
                else:
                    status.write("Visual Check Passed")

            # Display the Image
            st.image(base64.b64decode(image_b64), use_container_width=True)
            status.update(label="Process Complete", state="complete", expanded=False)
            
            with st.expander("View Final Prompt Details"):
                st.json(prompt_data.model_dump())

            st.session_state.local_history.append(types.Content(
                role="user", parts=[types.Part(text=user_input)]
            ))
            st.session_state.local_history.append(types.Content(
                role="model", parts=[types.Part(text=response.text)]
            ))
            pipeline.save_memory(st.session_state.local_history)

        except Exception as e:
            st.error(f"Workflow Error: {e}")