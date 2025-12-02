import os
import sys
import yaml
import base64
import json
import requests
import time
from google import genai
from google.genai import types
from pydantic import BaseModel

# ================= DATA STRUCTURES =================
class SDXLPrompt(BaseModel):
    positive: str
    negative: str

# ================= CONFIG LOADER =================
def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r", encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("[ERROR] config.yaml not found.")
        sys.exit(1)

# Initialize Config & Clients globally so they can be imported
config = load_config()

try:
    google_client = genai.Client(api_key=config['api_keys']['google'])
    STABILITY_KEY = config['api_keys']['stability']
except KeyError as e:
    print(f"[ERROR] Missing API Key: {e}")
    sys.exit(1)

# ================= MEMORY MANAGEMENT =================
def load_memory():
    """Loads history from JSON file into Gemini Content objects."""
    if not config['memory']['enabled']:
        return []

    file_path = config['memory']['file_path']
    
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        if not raw_data:
            return []
            
        history = []
        for item in raw_data:
            history.append(types.Content(
                role=item['role'],
                parts=[types.Part(text=part) for part in item['parts']]
            ))
        return history

    except Exception as e:
        print(f"[WARNING] Memory load failed: {e}")
        return []

def save_memory(history_list):
    """Saves a list of Content objects to JSON (FIFO)."""
    if not config['memory']['enabled']:
        return

    file_path = config['memory']['file_path']
    max_depth = config['memory'].get('max_history_depth', 10)

    # FIFO Logic
    if len(history_list) > max_depth:
        history_to_save = history_list[-max_depth:]
    else:
        history_to_save = history_list

    # Serialize
    serializable_history = []
    for content in history_to_save:
        parts_text = []
        for p in content.parts:
            if hasattr(p, 'text') and p.text:
                parts_text.append(p.text)
            elif isinstance(p, dict) and 'text' in p:
                parts_text.append(p['text'])
                
        serializable_history.append({
            "role": content.role,
            "parts": parts_text
        })

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save memory: {e}")

def clear_memory_file():
    """Helper to delete the memory file."""
    if os.path.exists(config['memory']['file_path']):
        os.remove(config['memory']['file_path'])

# ================= SDXL GENERATION =================
def generate_image_sdxl(prompts: SDXLPrompt):
    """Calls Stability API and returns the Base64 string of the image."""
    engine_id = config['models']['painter']
    url = f"https://api.stability.ai/v1/generation/{engine_id}/text-to-image"

    body = {
        "steps": config['generation']['steps'],
        "width": config['generation']['width'],
        "height": config['generation']['height'],
        "seed": 0, 
        "cfg_scale": config['generation']['cfg_scale'],
        "samples": 1,
        "text_prompts": [
            {"text": prompts.positive, "weight": 1},
            {"text": prompts.negative, "weight": -1}
        ],
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STABILITY_KEY}",
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        if response.status_code != 200:
            return None, f"API Error: {response.text}"

        data = response.json()
        # Return the raw base64 string so the UI can decide how to display/save it
        return data["artifacts"][0]["base64"], None
    except Exception as e:
        return None, str(e)