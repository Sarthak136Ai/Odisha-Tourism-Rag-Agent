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

# Bootstrap database if empty on startup
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
