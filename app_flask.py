import os
import time
import logging
from flask import Flask, request, jsonify, render_template, send_file, Response
import json
import requests

import config
from scraper import OdishaTourismScraper
from vector_store import OdishaVectorStore
from rag_pipeline import OdishaRAGPipeline
from utils.pdf_generator import generate_itinerary_pdf

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__, template_folder="templates")

# Initialize backend models and controllers
scraper = OdishaTourismScraper()
vector_store = OdishaVectorStore()
rag_pipeline = OdishaRAGPipeline()

# Bootstrap database if empty on startup (disabled on Vercel to prevent illegal writes and startup timeouts)
if not os.environ.get("VERCEL"):
    try:
        existing = vector_store.db.get()
        if not existing or len(existing.get("documents", [])) == 0:
            logger.info("Vector database is empty. Running initial bootstrapping on startup...")
            scraper.run_all_acquisition()
            vector_store.build_database()
    except Exception as e:
        logger.warning(f"Error checking DB size on startup: {e}. Running bootstrapping...")
        scraper.run_all_acquisition()
        vector_store.build_database()
else:
    logger.info("Vercel environment detected. Skipping startup database bootstrapping and checks.")


@app.route("/", methods=["GET"])
def index():
    """
    Serves the main single-page web dashboard.
    """
    return render_template("index.html", languages=config.LANGUAGES)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Accepts RAG chat query and streams the real-time response with citations.
    """
    data = request.json or {}
    query = data.get("query", "").strip()
    model = data.get("model", config.DEFAULT_LLM_MODEL)
    language = data.get("language", "en")

    if not query:
        return jsonify({"error": "Empty query"}), 400

    logger.info(f"RAG Request received for streaming: '{query}' [Model: {model}, Lang: {language}]")
    
    try:
        # 1. Context retrieval (Fast, takes ~0.1s)
        context, citations = rag_pipeline.retrieve_context(query, language=language)
        
        def generate():
            # Send citations first as a special SSE event
            yield f"event: citations\ndata: {json.dumps(citations)}\n\n"
            
            # Now stream tokens as they arrive from local Ollama
            full_answer = []
            start_time = time.time()
            for token in rag_pipeline.query_llm_with_context_stream(query, context, model, language):
                full_answer.append(token)
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                
            duration = time.time() - start_time
            # Send final completion event
            yield f"event: done\ndata: {json.dumps({'duration': round(duration, 2), 'answer': ''.join(full_answer)})}\n\n"
            
        return Response(generate(), mimetype="text/event-stream")
        
    except Exception as e:
        logger.error(f"Error executing RAG pipeline stream: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/itinerary", methods=["POST"])
def api_itinerary():
    """
    Generates a structured Day-by-Day travel itinerary.
    """
    data = request.json or {}
    destination = data.get("destination", "Puri & Konark (Spiritual & Heritage)").strip()
    duration = int(data.get("duration", 3))
    pace = data.get("pace", "balanced").strip()

    logger.info(f"Itinerary Request received: {destination} for {duration} days ({pace} pace)")

    try:
        itinerary_text = rag_pipeline.generate_custom_itinerary(
            location=destination,
            duration_days=duration,
            pace=pace
        )
        return jsonify({"itinerary": itinerary_text})
    except Exception as e:
        logger.error(f"Error generating itinerary: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/pdf", methods=["POST"])
def api_pdf():
    """
    Converts raw itinerary text into a styled PDF and returns it for download.
    """
    data = request.json or {}
    itinerary_text = data.get("itinerary", "").strip()
    destination = data.get("destination", "Odisha").strip()
    duration = data.get("duration", "3 Days").strip()

    if not itinerary_text:
        return jsonify({"error": "Empty itinerary text"}), 400

    pdf_filename = f"{destination.replace(' ', '_')}_itinerary.pdf"
    pdf_path = os.path.join(config.DATA_DIR, pdf_filename)

    logger.info(f"Generating PDF for destination: {destination}")

    try:
        success = generate_itinerary_pdf(
            itinerary_text=itinerary_text,
            filename=pdf_path,
            duration_days=duration,
            traveler_name="Odisha Explorer"
        )
        if success and os.path.exists(pdf_path):
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=pdf_filename,
                mimetype="application/pdf"
            )
        else:
            return jsonify({"error": "PDF Generation failed"}), 500
    except Exception as e:
        logger.error(f"Error generating PDF file: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/translate", methods=["POST"])
def api_translate():
    """
    Translates input text from any source language to a target language.
    """
    data = request.json or {}
    text = data.get("text", "").strip()
    source_lang = data.get("source_lang", "English")
    target_lang = data.get("target_lang", "Odia")
    
    if not text:
        return jsonify({"error": "Empty text"}), 400
        
    logger.info(f"Translation Request received: '{text}' from {source_lang} to {target_lang}")
    
    try:
        # Prompt model to perform exact translation with highly robust script guard rails
        system_prompt = (
            f"You are a highly precise multi-lingual translator. Translate the text exactly from {source_lang} to {target_lang}. "
            "Return ONLY the direct, plain translated text. Do not add any conversational text, "
            "explanations, notes, or extra markup.\n"
        )
        if target_lang.lower() in ["odia", "oriya", "or"]:
            system_prompt += (
                "CRITICAL INSTRUCTION FOR NATIVE ODIA:\n"
                "1. You MUST translate strictly into normal, everyday Odia language, written entirely in native Odia script (ଓଡ଼ିଆ ଅକ୍ଷର). Do NOT use English/Latin alphabet, phonetic transliteration, or Devanagari.\n"
                "2. ABSOLUTE BAN ON HINDI/BENGALI MIXTURE: You must NEVER mix Bengali or Hindi grammar/words into Odia. Specifically:\n"
                "   - NEVER use equivalents of the Hindi word 'sabse' (for 'best/most'). You MUST use the Odia word 'ସବୁଠୁ' (sabuthu) or 'ସବୁଠାରୁ' (e.g., 'ସବୁଠୁ ଭଲ' (sabuthu bhala)).\n"
                "   - NEVER use equivalents of the Bengali word 'bhalo' (for 'good/well'). You MUST use the Odia word 'ଭଲ' (bhala).\n"
                "   - NEVER use equivalents of the Bengali word 'ei' (for 'this'). You MUST use the Odia word 'ଏହି' (ehi) or 'ଏ' (e).\n"
                "   - NEVER use equivalents of 'keno' or 'kene' (for 'why/what'). You MUST use the Odia word 'କଣ' (kana) or 'କାହିଁକି' (kahinki).\n"
                "3. Keep it in normal, casual Odia as real people speak in daily life. Avoid overly pure, formal, literary, or Sanskritized vocabulary. For example, do NOT use 'ଖାଦ୍ୟ' (khadya) for food; use the everyday colloquial word 'ଖାଇବା' (khaiba) instead.\n"
                "4. Ensure correct translation of homonyms like 'raga' (which can mean both anger and spicy flavor in Odia). Specifically:\n"
                "   - 'I am angry' -> 'ମୁଁ ରାଗିଛି' or 'ମୋତେ ରାଗ ଲାଗୁଛି' (Never translate literally to eating anger like 'ମୁଁ ରାଗ ଖାଇଛି')\n"
                "   - 'The food is spicy' -> 'ଖାଇବା ବହୁତ ରାଗ' or 'ଏହି ଖାଇବା ବହୁତ ରାଗ'\n"
                "5. It is perfectly natural to use common English loan words written in Odia script if everyday Odia speakers use them in normal conversations (e.g. 'ବସ', 'ଟ୍ରେନ', 'ଟିକେଟ', 'ଗେଟ', 'ଟାଇମ', 'ହୋଟେଲ', 'ଫୋନ', 'ଫଟୋ', 'ୱାଟର', 'ହେଲ୍ପ').\n"
                "6. Reference translations for common sentences to match normal daily talking:\n"
                "   - 'good morning' -> 'ଶୁଭ ସକାଳ' (Do NOT translate to 'ଶୁଭ ଦ୍ଵିପ୍ରହର')\n"
                "   - 'good afternoon' -> 'ଶୁଭ ଦ୍ଵିପ୍ରହର'\n"
                "   - 'good evening' -> 'ଶୁଭ ସନ୍ଧ୍ୟା'\n"
                "   - 'good night' -> 'ଶୁଭ ରାତ୍ରି'\n"
                "   - 'I am angry' -> 'ମୁଁ ରାଗିଛି'\n"
                "   - 'What is the best food in this area?' -> 'ଏହି ଜାଗାର ସବୁଠୁ ଭଲ ଖାଇବା କଣ?'\n"
                "   - 'hii, this is Sarthak, what's your name?' -> 'ହାଏ, ମୁଁ ସାର୍ଥକ, ତୁମ ନାଁ କଣ?'\n"
                "   - 'what is your name?' -> 'ତୁମ ନାଁ କଣ?' or 'ଆପଣଙ୍କ ନାଁ କଣ?'\n"
                "   - 'my name is ...' -> 'ମୋର ନାଁ ...'\n"
                "   - 'how are you?' -> 'ତୁମେ କେମିତି ଅଛ?' or 'ଆପଣ କେମିତି ଅଛନ୍ତି?'\n"
                "   - 'where is the temple?' -> 'ମନ୍ଦିର କେଉଁଠି ଅଛି?'\n"
                "   - 'thank you' -> 'ଧନ୍ୟବାଦ'\n"
                "   - 'did you eat?' -> 'ତୁମେ ଖାଇଲ କି?' or 'ଖାଇଲ କି ନାହିଁ?'\n"
                "   - 'yes, I ate' -> 'ହଁ, ଖାଇଲି'\n"
                "   - 'where can I get food?' -> 'ମୋତେ ଖାଇବା କେଉଁଠି ମିଳିବ?'"
            )
        elif target_lang.lower() == "phonetic odia":
            system_prompt += (
                "CRITICAL INSTRUCTION FOR PHONETIC ODIA:\n"
                "1. You MUST translate strictly into normal, everyday Odia language, written entirely in the English/Latin alphabet phonetically (transliterated WhatsApp chat style). Do NOT use the native Odia script (ଓଡ଼ିଆ) or Devanagari.\n"
                "2. ABSOLUTE BAN ON HINDI/BENGALI MIXTURE: You must NEVER mix Bengali or Hindi grammar/words into Odia. Specifically:\n"
                "   - NEVER use the Hindi word 'sabse' (for 'best/most'). You MUST use the Odia word 'sabuthu' or 'sabutu' (e.g., 'sabuthu bhala' instead of 'sabse bhalo').\n"
                "   - NEVER use the Bengali word 'bhalo' (for 'good/well'). You MUST use the Odia word 'bhala'.\n"
                "   - NEVER use the Bengali word 'ei' (for 'this'). You MUST use the Odia word 'ehi' or 'e'.\n"
                "   - NEVER use 'keno' or 'kene' (for 'why/what'). You MUST use the Odia word 'kana' or 'kahinki'.\n"
                "3. Keep it in normal, casual Odia as real people speak in daily life. Avoid overly pure, formal, literary, or Sanskritized vocabulary. For example, do NOT use 'khadya' for food; use the everyday colloquial word 'khaiba' instead.\n"
                "4. Ensure correct translation of homonyms like 'raga/raag' (which can mean both anger and spicy flavor in Odia). Specifically:\n"
                "   - 'I am angry' -> 'Mu ragichhi' or 'Mote raga laguchi' (Never translate literally to eating anger like 'Mu raga khaichi' or 'Mu raag khaichi')\n"
                "   - 'The food is spicy' -> 'Khaiba bahut raga' or 'Ehi khaiba bahut raga'\n"
                "5. It is perfectly natural to use common English loan words if everyday Odia speakers use them in normal conversations (e.g. 'bus', 'train', 'ticket', 'gate', 'time', 'hotel', 'phone', 'photo', 'water', 'help').\n"
                "6. Reference translations for common sentences to match normal WhatsApp-style talking:\n"
                "   - 'good morning' -> 'Subha sakala' (Do NOT translate to 'Subha dhupare' or 'Subha dwipahara')\n"
                "   - 'good afternoon' -> 'Subha dwipahara' or 'Subha dhupare'\n"
                "   - 'good evening' -> 'Subha sandhya'\n"
                "   - 'good night' -> 'Subha ratri'\n"
                "   - 'I am angry' -> 'Mu ragichhi'\n"
                "   - 'What is the best food in this area?' -> 'Ehi jagara sabuthu bhala khaiba kana?' (Do NOT write 'Ei jaga ra sabse bhalo khaiba kana?')\n"
                "   - 'hii, this is Sarthak, what's your name?' -> 'Hi, mu Sarthak, tuma naa kana?'\n"
                "   - 'what is your name?' -> 'Tuma naa kana?' or 'Apananka naa kana?'\n"
                "   - 'my name is ...' -> 'Mora naa ...'\n"
                "   - 'how are you?' -> 'Tume kemiti achha?' or 'Apana kemiti achhanti?'\n"
                "   - 'where is the temple?' -> 'Mandira keunthi achhi?'\n"
                "   - 'thank you' -> 'Dhanyabad' or 'Thank you'\n"
                "   - 'did you eat?' -> 'Tume khaila ki?' or 'Khaila ki nahi?'\n"
                "   - 'yes, I ate' -> 'Han, khaili'\n"
                "   - 'where can I get food?' -> 'Mote khaiba keunthi miliba?'"
            )

        gemini_api_key = getattr(config, "GEMINI_API_KEY", "")
        if gemini_api_key:
            logger.info("Using Google Gemini API for highly precise translation.")
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
                gemini_payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": text}
                            ]
                        }
                    ],
                    "systemInstruction": {
                        "parts": [
                            {"text": system_prompt}
                        ]
                    },
                    "generationConfig": {
                        "temperature": 0.0
                    }
                }
                response = requests.post(url, json=gemini_payload, timeout=12)
                if response.status_code == 200:
                    result = response.json()
                    candidates = result.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            translated_text = parts[0].get("text", "").strip()
                            logger.info("Google Gemini translation succeeded.")
                            return jsonify({"translation": translated_text})
                
                logger.error(f"Gemini API returned error: {response.status_code} - {response.text}. Falling back to Groq...")
            except Exception as gemini_err:
                logger.error(f"Gemini API execution error: {gemini_err}. Falling back to Groq...")

        # Fallback / Default to Groq Llama Translation (using the highly capable Llama 3.3 70B model)
        logger.info("Using Groq API fallback for translation with llama-3.3-70b-versatile.")
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {"role": "user", "content": text}
            ],
            "temperature": 0.0,
            "stream": False
        }
        
        response = requests.post(
            rag_pipeline.groq_url,
            headers=rag_pipeline.headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            choices = result.get("choices", [])
            if choices:
                translated_text = choices[0].get("message", {}).get("content", "").strip()
                return jsonify({"translation": translated_text})
        
        return jsonify({"error": f"Failed to translate. Status code: {response.status_code}"}), 500
    except Exception as e:
        logger.error(f"Error executing translation API: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/diagnostics", methods=["GET"])
def api_diagnostics():
    """
    Checks Groq API connectivity and Vector DB document counts.
    """
    groq_active = False
    supported_models = config.SUPPORTED_LLM_MODELS
    
    try:
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            groq_active = True
    except Exception:
        pass
        
    db_document_count = 0
    try:
        db_data = vector_store.db.get()
        if db_data and "documents" in db_data:
            db_document_count = len(db_data["documents"])
    except Exception:
        pass

    return jsonify({
        "ollama_active": groq_active,  # Kept key as 'ollama_active' for front-end template compatibility
        "active_model": config.DEFAULT_LLM_MODEL,
        "supported_models": supported_models,
        "db_document_count": db_document_count
    })


@app.route("/api/reindex", methods=["POST"])
def api_reindex():
    """
    Triggers web scrapes & PDF parses to update the local ChromaDB database.
    """
    logger.info("Database rebuild API triggered.")
    try:
        scraper.run_all_acquisition()
        vector_store.build_database(force_rebuild=True)
        
        # Get updated count
        db_data = vector_store.db.get()
        db_document_count = len(db_data["documents"]) if db_data else 0
        
        return jsonify({
            "success": True,
            "message": "Vector Database successfully rebuilt!",
            "db_document_count": db_document_count
        })
    except Exception as e:
        logger.error(f"Error rebuilding database: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting local Flask web server...")
    # Bind to localhost on port 5000 (standard local development port)
    app.run(host="127.0.0.1", port=5000, debug=True)
