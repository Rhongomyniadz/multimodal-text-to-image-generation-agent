import streamlit as st
import base64
import time
from google.genai import types

# Import your logic file
import pipeline

# Page Config
st.set_page_config(page_title="AI Art Studio", page_icon="🎨", layout="wide")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Settings")
    st.text(f"Brain: {pipeline.config['models']['brain']}")
    st.text(f"Painter: {pipeline.config['models']['painter']}")
    
    if st.button("🧹 Clear Memory", type="primary"):
        # Call the function from pipeline.py
        pipeline.clear_memory_file()
        st.session_state.local_history = []
        st.success("Memory wiped!")
        time.sleep(1)
        st.rerun()

# ================= MAIN INTERFACE =================
st.title("🎨 Context-Aware Art Generator")

# Initialize Session State using pipeline function
if "local_history" not in st.session_state:
    st.session_state.local_history = pipeline.load_memory()

# 1. Display Chat History
# We iterate through the loaded history to show previous turn
for msg in st.session_state.local_history:
    with st.chat_message(msg.role):
        # Extract text from parts for display
        text_content = ""
        for p in msg.parts:
            if hasattr(p, 'text'): text_content += p.text
            elif isinstance(p, dict): text_content += p.get('text', '')
        
        # If it looks like JSON (Model response), make it pretty
        if msg.role == "model" and "{" in text_content:
            st.code(text_content, language="json")
        else:
            st.write(text_content)

# 2. Handle New Input
if user_input := st.chat_input("Describe your idea..."):
    
    # Display User Message
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

    # Create Chat Object using imported client
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
        status = st.status("Agent is optimizing prompt...", expanded=True)
        
        try:
            # --- CALL GEMINI ---
            response = chat.send_message(user_input)
            prompt_data = response.parsed
            
            status.write("Prompt Generated!")
            st.code(response.text, language="json")

            # --- CALL SDXL (Using pipeline function) ---
            status.update(label="Generative AI Painting...", state="running")
            
            image_b64, error_msg = pipeline.generate_image_sdxl(prompt_data)
            
            if error_msg:
                st.error(error_msg)
                status.update(label="Generation Failed", state="error")
            else:
                # Display Image
                st.image(base64.b64decode(image_b64), use_container_width=True)
                status.update(label="Image Ready", state="complete", expanded=False)

            # --- UPDATE MEMORY (Using pipeline function) ---
            # Manually append to local state
            st.session_state.local_history.append(types.Content(
                role="user", parts=[types.Part(text=user_input)]
            ))
            st.session_state.local_history.append(types.Content(
                role="model", parts=[types.Part(text=response.text)]
            ))
            
            # Save to disk
            pipeline.save_memory(st.session_state.local_history)

        except Exception as e:
            st.error(f"Interaction Error: {e}")