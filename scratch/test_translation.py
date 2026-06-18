import requests
import json

def test_translation():
    url = "http://127.0.0.1:5000/api/translate"
    payload = {
        "text": "What is the best food in this area?",
        "source_lang": "English",
        "target_lang": "Odia"
    }
    
    try:
        response = requests.post(url, json=payload)
        data = response.json()
        print("Status Code:", response.status_code)
        
        translation = data.get("translation", "")
        # Save output in utf-8 file to prevent cp1252 encoding issues on windows console
        with open("scratch/test_translation_output.txt", "w", encoding="utf-8") as f:
            f.write(translation)
            
        print("Translation saved to scratch/test_translation_output.txt successfully!")
        
        assert "---" in translation, "Translation should contain the splitter '---'"
        parts = translation.split("---")
        assert len(parts) == 2, "Translation should be divided into exactly two parts"
        print("Native script part len:", len(parts[0].strip()))
        print("Phonetic transliteration part:", parts[1].strip())
        print("\nTest PASSED successfully!")
    except Exception as e:
        print("Test FAILED:", e)

if __name__ == "__main__":
    test_translation()
