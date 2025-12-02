# Multimodal Text-to-Image Agent (LLM + Image Gen) — Prompt Pipeline + Multi-turn Memory + Tool Routing

An **agentic text-to-image assistant** that turns vague user requests into **production-ready English prompts**, supports **multi-turn edits** (e.g., “make it red”, “change to sunset”) without drifting off-topic, and routes requests to the right tool (draw vs. Q&A) to control cost.

Optional: a **visual feedback loop** where a VLM “looks at” the generated image, checks missing requirements, and automatically revises the prompt and regenerates.

---

## Demo Features

### ✅ Required (Must-have)
- **Prompt Engineering Pipeline**
  - Input: vague natural language
  - Output: high-quality English prompt formatted for a target model family (SDXL / Midjourney / DALL·E 3)
  - Includes: quality, lighting, composition, camera terms, plus model params.

- **Context Awareness (Multi-turn Memory)**
  - Maintains session state: previous PromptSpec + constraints checklist
  - Supports incremental edits: “把它改成红色” updates previous prompt rather than rewriting a random new scene.

- **API Orchestration + Web UI**
  - UI: Gradio / Streamlit
  - Flow: user input → agent rewrites prompt (shown in UI) → image generation → show image
  - Terminal logs: show routing, prompt versioning, tool calls.

### ⭐ Optional (Nice-to-have)
- **Visual Feedback Loop**
  - After image generation, a VLM checks whether key constraints appear in the image.
  - If missing, the agent revises the prompt and regenerates (limited retries).
- **Function Calling / Routing**
  - If user asks “how to write prompts”, the agent answers in text only (no image API cost).
  - If user says “generate an image”, it calls image generation.

---

## Architecture Overview

```mermaid
flowchart LR
  UI[Gradio/Streamlit UI] --> ORCH[Agent Orchestrator]
  ORCH --> ROUTER[Intent Router]

  ROUTER -->|DRAW/EDIT| PE[Prompt Engineer LLM]
  ROUTER -->|QA| QA[Text Answer LLM]

  PE <--> MEM[Session Memory (PromptSpec + Checklist)]
  PE --> ADAPT[Model Adapter (SDXL/MJ/DALL·E3)]
  ADAPT --> IMG[Image Generation API]
  IMG --> OUT[Image + Metadata]

  OUT -->|optional| VLM[VLM Validator]
  VLM -->|revise| PE

  ORCH --> LOG[JSONL Logs]