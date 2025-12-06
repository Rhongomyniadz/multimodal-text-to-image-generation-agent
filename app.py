import streamlit as st
import base64
import time
from google.genai import types

# Import pipeline and logging
import pipeline
import vlm_feedback 

# Use the logger configured in pipeline
logger = pipeline.logger

st.set_page_config(page_title="AI Art Studio", page_icon="🎨", layout="wide")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("Settings")
    
    enable_feedback = st.checkbox(
        "Enable Visual Feedback", 
        value=pipeline.config.get('visual_feedback', {}).get('enabled', True)
    )
    pipeline.config['visual_feedback']['enabled'] = enable_feedback

    if st.button("Clear Memory", type="primary"):
        pipeline.clear_memory_file()
        st.session_state.local_history = []
        st.success("Memory wiped!")
        logger.info("[SYSTEM] User manually cleared memory.")
        time.sleep(1)
        st.rerun()

# ================= MAIN INTERFACE =================
st.title("Context-Aware Art Generator")

# Load History
if "local_history" not in st.session_state:
    st.session_state.local_history = pipeline.load_memory()

# Display Chat History
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

# Handle Input
if user_input := st.chat_input("Describe your idea..."):
    
    # Log User Input
    logger.info(f"\n[USER] Input: '{user_input}'")
    
    with st.chat_message("user"):
        st.write(user_input)

    # System Prompt
    SYSTEM_INSTRUCTION = """
    You are an expert SDXL Prompt Engineer with persistent memory.
    ALWAYS check history. Output structured JSON {positive, negative}.
    """
    
    # Create Chat
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
            # === ATTEMPT 1: Generate ===
            status.write("Generating Prompt...")
            logger.info("[BRAIN] Analyzing context and optimizing prompt...")
            
            response = chat.send_message(user_input)
            prompt_data = response.parsed
            
            logger.info(f"[BRAIN] Optimization Complete.")
            logger.info(f"   -> Original: '{user_input}'")
            logger.info(f"   -> Optimized: '{prompt_data.positive[:60]}...'")

            status.write("Painting (Attempt 1)...")
            image_b64, error_msg = pipeline.generate_image_sdxl(prompt_data)
            
            if error_msg:
                st.error(error_msg)
                st.stop()

            # === VISUAL FEEDBACK LOOP (VLM) ===
            if pipeline.config['visual_feedback']['enabled']:
                status.write("VLM: Inspecting image...")
                logger.info("[CRITIC] Starting Visual Inspection...")
                
                critique = vlm_feedback.analyze_image(image_b64, user_input)
                
                if not critique.passed:
                    logger.warning(f"[CRITIC] FAIL: {critique.reason}. Missing: {critique.missing_elements}")
                    status.warning(f"Issue: {critique.reason}")
                    status.write(f"Auto-fixing...")
                    
                    # Correction Prompt
                    correction_prompt = f"""
                    SYSTEM ALERT: The previous image failed visual inspection.
                    Reason: {critique.reason}.
                    Missing Elements: {critique.missing_elements}.
                    TASK: Regenerate the SDXL JSON prompt. 
                    Keep the style, but STRONG EMPHASIS on including: {critique.missing_elements}.
                    """
                    
                    logger.info("[BRAIN] Applying fix and regenerating prompt...")
                    response_fix = chat.send_message(correction_prompt)
                    prompt_data_fix = response_fix.parsed
                    
                    status.write("Painting (Attempt 2 - Fixed)...")
                    image_b64_fix, error_msg_fix = pipeline.generate_image_sdxl(prompt_data_fix)
                    
                    if not error_msg_fix:
                        image_b64 = image_b64_fix
                        prompt_data = prompt_data_fix
                        response = response_fix
                        status.success("Fixed and Regenerated!")
                        logger.info("[SYSTEM] Auto-fix successful.")
                    else:
                        status.error("Retry failed, using original.")
                        logger.error("[SYSTEM] Auto-fix generation failed.")
                else:
                    status.write("Visual Check Passed")
                    logger.info("[CRITIC] PASS: Image looks correct.")

            # === DISPLAY & SAVE ===
            st.image(base64.b64decode(image_b64), use_container_width=True)
            status.update(label="Complete", state="complete", expanded=False)
            
            with st.expander("Prompt Details"):
                st.json(prompt_data.model_dump())

            # Save to Memory
            st.session_state.local_history.append(types.Content(
                role="user", parts=[types.Part(text=user_input)]
            ))
            st.session_state.local_history.append(types.Content(
                role="model", parts=[types.Part(text=response.text)]
            ))
            pipeline.save_memory(st.session_state.local_history)

        except Exception as e:
            logger.error(f"[SYSTEM] Workflow Error: {e}")
            st.error(f"Workflow Error: {e}")