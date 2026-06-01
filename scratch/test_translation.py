import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import json
import config

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {config.GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def test_translation(prompt):
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": prompt
            },
            {"role": "user", "content": "Hii this is Sarthak, what's your name?"}
        ],
        "temperature": 0.0,
        "stream": False
    }
    
    for attempt in range(3):
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=10)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"].strip()
            elif res.status_code == 429:
                print(f"Rate limited (429), waiting 5s... Attempt {attempt+1}")
                time.sleep(5)
            else:
                return f"Error {res.status_code}: {res.text}"
        except Exception as e:
            print(f"Connection error: {e}. Retrying in 2s...")
            time.sleep(2)
    return "Failed after multiple attempts"

print("Running translations...")
res1 = test_translation("You are a highly precise translator. Translate the text exactly from English to Odia. Return ONLY the direct translated text. Do not add any conversational text, explanations, notes, or extra tags. If translating to Odia, output in proper Odia script.")
res2 = test_translation(
    "You are a highly precise multi-lingual translator. Translate the text exactly from English to Odia. "
    "Return ONLY the direct, plain translated text in the native script of the target language. Do not add any conversational text, "
    "explanations, notes, or extra markup.\n"
    "CRITICAL INSTRUCTION: The target language is Odia (ଓଡ଼ିଆ). You MUST translate to Odia and write strictly in the proper Odia script (ଓଡ଼ିଆ ଅକ୍ଷର, using characters like 'ନମସ୍କାର', 'ଓଡ଼ିଆ', 'ସାର୍ଥକ', 'ନାମ', 'କଣ'). "
    "WARNING: Do NOT under any circumstances output Gurmukhi/Punjabi script (characters like 'ਹ', 'ਆ', 'ਨ', 'ਕ') or Hindi/Devanagari script! Odia is a completely different language with its own distinct rounded alphabet script. Double-check that your output uses strictly Odia characters."
)

with open("scratch/output.txt", "w", encoding="utf-8") as f:
    f.write(f"TEST 1:\n{res1}\n\nTEST 2:\n{res2}\n")

print("Success writing to scratch/output.txt")
