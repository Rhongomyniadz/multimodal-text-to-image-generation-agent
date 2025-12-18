import streamlit as st
import base64
import time
from google.genai import types

import pipeline
import vlm_feedback 
import lc_workflow

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


@st.cache_resource
def _get_langchain_runnables():
    """Build LangChain runnables once per Streamlit server process."""
    return lc_workflow.build_runnables()


runnables = _get_langchain_runnables()

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

    with st.chat_message("model"):
        status = st.status("Agent is working...", expanded=True)
        
        try:
            state = {
                "user_input": user_input,
                "history": st.session_state.local_history,
                "enable_feedback": pipeline.config["visual_feedback"]["enabled"],
                "system_instruction": lc_workflow.DEFAULT_SYSTEM_INSTRUCTION,
            }

            # Brain
            status.write("Generating Prompt...")
            state = runnables["brain"].invoke(state)

            # Painter (Attempt 1)
            status.write("Painting (Attempt 1)...")
            state = runnables["paint"].invoke(state)

            # Critic + Auto-fix (Attempt 2)
            if pipeline.config["visual_feedback"]["enabled"]:
                status.write("VLM: Inspecting image...")
                state = runnables["critic"].invoke(state)
                if not state["critique"].passed:
                    status.warning(f"Issue: {state['critique'].reason}")
                    status.write("Auto-fixing...")

            state = runnables["auto_fix"].invoke(state)

            image_b64 = state["image_b64"]
            prompt_data = state["prompt_data"]
            response_text = state["brain_response_text"]

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
                role="model", parts=[types.Part(text=response_text)]
            ))
            pipeline.save_memory(st.session_state.local_history)

        except Exception as e:
            logger.error(f"[SYSTEM] Workflow Error: {e}")
            st.error(f"Workflow Error: {e}")
