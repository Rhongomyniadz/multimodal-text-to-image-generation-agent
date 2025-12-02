# 🎨 AI Art Director - Intelligent Art Pipeline

This is an end-to-end Intelligent Art Agent built on **Google Gemini (Brain/VLM)** and **Stability AI (SDXL)**. It goes beyond simple text-to-image generation, acting as a creative system with **long/short-term memory**, **context awareness**, and **visual self-correction** capabilities.

---

## 🏗️ System Architecture

The system follows a modular design consisting of three core roles: The Brain (Prompt Engineer), The Painter (Rendering Engine), and The Critic (Quality Assurance), all orchestrated via a Streamlit frontend.

```mermaid
graph TD
    User[👤 User] -->|1. Input "Draw a cyberpunk cat"| UI[🖥️ Streamlit Web UI]
    
    subgraph "🧠 Brain & Memory"
        UI -->|2. Request + History| Brain[🤖 Gemini 1.5 Pro (Brain)]
        Brain <-->|Read/Write Context| Memory[(💾 agent_memory.json)]
        Brain -->|3. Structured Prompt (JSON)| UI
    end
    
    subgraph "🎨 Painter"
        UI -->|4. Call Generation API| SDXL[🖼️ Stability AI (SDXL)]
        SDXL -->|5. Return Base64 Image| UI
    end
    
    subgraph "👁️ Visual Feedback Loop"
        UI -.->|6. (Optional) Image + User Request| VLM[🧐 Gemini VLM (Critic)]
        VLM -->|7. Visual Analysis| Check{🔍 Pass?}
        
        Check -- YES --> Display[✅ Display Final Image]
        Check -- NO (Missing elements) --> AutoFix[🛠️ Build Correction Prompt]
        AutoFix -->|8. Trigger Regenerate| Brain
    end

    style User fill:#f9f,stroke:#333,stroke-width:2px
    style Brain fill:#bbf,stroke:#333,stroke-width:2px
    style SDXL fill:#bfb,stroke:#333,stroke-width:2px
    style VLM fill:#fbb,stroke:#333,stroke-width:2px