import os

# --- BASE DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectordb")
MODELS_DIR = os.path.join(BASE_DIR, "models")
UTILS_DIR = os.path.join(BASE_DIR, "utils")

# Create folders if they do not exist (only if not running on Vercel to avoid read-only filesystem crash)
if not os.environ.get("VERCEL"):
    for folder in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR, MODELS_DIR, UTILS_DIR]:
        os.makedirs(folder, exist_ok=True)

# --- SCRAPER CONFIGURATION ---
# Reliable target URLs for scraping Odisha tourism info
SCRAPE_URLS = {
    "wikipedia_odisha": "https://en.wikipedia.org/wiki/Odisha",
    "wikipedia_tourism": "https://en.wikipedia.org/wiki/Tourism_in_Odisha",
    "wikipedia_konark": "https://en.wikipedia.org/wiki/Konark_Sun_Temple",
    "wikipedia_puri": "https://en.wikipedia.org/wiki/Jagannath_Temple,_Puri",
    "wikipedia_chilika": "https://en.wikipedia.org/wiki/Chilika_Lake",
    "unesco_konark": "https://whc.unesco.org/en/list/246"
}

# --- EMBEDDINGS & VECTORSTORE CONFIGURATION ---
# Using BAAI/bge-small-en-v1.5 local absolute directory path to prevent network/async conflicts
EMBEDDING_MODEL_NAME = os.path.join(MODELS_DIR, "bge-small-en-v1.5")
# Fallback model in case the machine has low resources
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Read GROQ key securely from environment variables, or import from untracked local_config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    try:
        import local_config
        GROQ_API_KEY = local_config.GROQ_API_KEY
    except ImportError:
        pass
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"
SUPPORTED_LLM_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

# Read GEMINI key securely from environment variables, or import from untracked local_config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    try:
        import local_config
        GEMINI_API_KEY = getattr(local_config, "GEMINI_API_KEY", "")
    except ImportError:
        pass



# --- UI & BRANDING ---
# Theme colors: Warm Golden Terracotta theme for Odisha Tourism
BRAND_COLORS = {
    "primary": "#D4AF37",       # Marigold Gold
    "secondary": "#E07A5F",     # Terracotta Orange
    "background": "#F4F1DE",    # Sand Cream
    "text": "#3D405B",          # Deep Charcoal
    "accent": "#81B29A",        # Soft Olive Green (Chilika Lake vibes)
    "dark_primary": "#2D3142",  # Dark Mode Background
    "dark_text": "#F4F1DE"
}

# Supported UI languages
LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "de": "Deutsch (German)",
    "fr": "Français (French)"
}
