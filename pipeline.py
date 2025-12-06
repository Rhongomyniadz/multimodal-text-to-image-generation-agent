import os
import sys
import yaml
import json
import requests
import time
import logging
from logging.handlers import RotatingFileHandler
from google import genai
from google.genai import types
from pydantic import BaseModel

# ================= LOGGING SETUP =================
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "pipeline.log")
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, LOG_FILE)

logger = logging.getLogger("AI_Pipeline")
logger.setLevel(logging.INFO)
logger.propagate = False  # avoid duplicate logs if root logger is configured elsewhere

fmt = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Clear existing handlers (useful if running in notebooks / reload)
if logger.handlers:
    logger.handlers.clear()

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(fmt)

# File handler (rotates to avoid infinite growth)
file_handler = RotatingFileHandler(
    filename=log_path,
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=3,
    encoding="utf-8",
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"Logging to console and file: {log_path}")

# ================= DATA STRUCTURES =================
class SDXLPrompt(BaseModel):
    positive: str
    negative: str

# ================= CONFIG LOADER =================
def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            logger.info(f"Loading configuration from {config_path}...")
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error("config.yaml not found.")
        sys.exit(1)

config = load_config()

try:
    google_client = genai.Client(api_key=config["api_keys"]["google"])
    STABILITY_KEY = config["api_keys"]["stability"]
    logger.info("API Clients initialized successfully.")
except KeyError as e:
    logger.error(f"Missing API Key: {e}")
    sys.exit(1)

# ================= MEMORY MANAGEMENT =================
def load_memory():
    """Loads history from JSON file."""
    if not config["memory"]["enabled"]:
        return []

    file_path = config["memory"]["file_path"]
    if not os.path.exists(file_path):
        logger.info("No existing memory file found. Starting fresh.")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not raw_data:
            return []

        history = []
        for item in raw_data:
            history.append(
                types.Content(
                    role=item["role"],
                    parts=[types.Part(text=part) for part in item["parts"]],
                )
            )
        logger.info(f"Loaded {len(history)} messages from memory.")
        return history
    except Exception as e:
        logger.warning(f"Memory load failed: {e}")
        return []

def save_memory(history_list):
    """Saves memory to JSON (FIFO)."""
    if not config["memory"]["enabled"]:
        return

    file_path = config["memory"]["file_path"]
    max_depth = config["memory"].get("max_history_depth", 10)

    history_to_save = history_list[-max_depth:] if len(history_list) > max_depth else history_list

    serializable_history = []
    for content in history_to_save:
        parts_text = []
        for p in content.parts:
            if hasattr(p, "text") and p.text:
                parts_text.append(p.text)
            elif isinstance(p, dict) and "text" in p:
                parts_text.append(p["text"])

        serializable_history.append({"role": content.role, "parts": parts_text})

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_history, f, ensure_ascii=False, indent=2)
        logger.info(f"Memory saved to {file_path} ({len(history_to_save)} items).")
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")

def clear_memory_file():
    if os.path.exists(config["memory"]["file_path"]):
        os.remove(config["memory"]["file_path"])
        logger.info("Memory file deleted.")

def generate_image_sdxl(prompts: SDXLPrompt):
    """Calls Stability API."""
    engine_id = config["models"]["painter"]
    url = f"https://api.stability.ai/v1/generation/{engine_id}/text-to-image"

    logger.info("=" * 40)
    logger.info("[PAINTER] Received Optimized Prompt Request")
    logger.info(f"[PAINTER] Engine: {engine_id}")
    logger.info(f"[PAINTER] Positive: {prompts.positive[:100]}... (truncated)")
    logger.info("=" * 40)

    body = {
        "steps": config["generation"]["steps"],
        "width": config["generation"]["width"],
        "height": config["generation"]["height"],
        "seed": 0,
        "cfg_scale": config["generation"]["cfg_scale"],
        "samples": 1,
        "text_prompts": [
            {"text": prompts.positive, "weight": 1},
            {"text": prompts.negative, "weight": -1},
        ],
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STABILITY_KEY}",
    }

    try:
        logger.info("[PAINTER] Sending request to Stability AI...")
        start_time = time.time()
        response = requests.post(url, headers=headers, json=body)
        duration = time.time() - start_time

        if response.status_code != 200:
            logger.error(f"[PAINTER] API Error ({response.status_code}): {response.text}")
            return None, f"API Error: {response.text}"

        logger.info(f"[PAINTER] Image generation successful ({duration:.2f}s)")
        data = response.json()
        return data["artifacts"][0]["base64"], None
    except Exception as e:
        logger.error(f"[PAINTER] Connection Error: {e}")
        return None, str(e)