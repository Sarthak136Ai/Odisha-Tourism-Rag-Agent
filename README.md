# 🕌 AI-Powered Local RAG Odisha Tourism Web App (Flask)

A complete, fully offline, and highly secure Retrieval-Augmented Generation (RAG) Tourist Agent focused on **Odisha Tourism**. 

Built using **Python 3.11**, **Flask**, **LangChain**, **ChromaDB**, and **Ollama**, this agent allows users to query, explore, and map out itineraries for Odisha's rich heritage sites, beaches, temples, cuisines, and nature preserves. The system does not depend on cloud-based APIs like OpenAI or Gemini—it runs **100% locally** on your machine.

---

## 📐 System Architecture

```
                 +--------------------------+
                 |   Internet Web Pages     |  <-- Scrapes Wikipedia, Official Tourism,
                 |   & Local PDF Documents  |      UNESCO, Heritage Blogs & pdf files.
                 +--------------------------+
                              |
                              v
                 +--------------------------+
                 |   BeautifulSoup Scraper  |  <-- Extracts clean text and removes ads,
                 |     & PDF Text Parser    |      navigations, and sidebars.
                 +--------------------------+
                              |
                              v
                 +--------------------------+
                 |  Recursive Text Splitter |  <-- Chunk size: 1000 characters,
                 |   (preserving contexts)  |      Overlap size: 200 characters.
                 +--------------------------+
                              |
                              v
                 +--------------------------+
                 |   HuggingFace Embeddings |  <-- BAAI/bge-small-en-v1.5 model loaded
                 |     (Local SentenceTF)   |      locally (auto GPU/CPU execution).
                 +--------------------------+
                              |
                              v
                 +--------------------------+
                 |    Chroma Vector DB      |  <-- Persistent local directory storage
                 |   (with deduplication)   |      with fast similarity indexing.
                 +--------------------------+
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
        +-------------------+   +--------------------+
        | Vector Retriever  |   | Local Ollama Server|  <-- Restricts prompts to context
        |  (Semantic Search)|   | (Phi3/Llama3 Model)|      to eliminate hallucinations.
        +-------------------+   +--------------------+
                  |                       |
                  +-----------+-----------+
                              |
                              v
                 +--------------------------+
                 |    Flask Web Server      |  <-- Serves single-page HTML, API endpoints,
                 |   (Templates & Fetch JS) |      Voice guide stream, and PDF itinerary.
                 +--------------------------+
```

---

## ✨ Features

1. **Intelligent Web Scraping & PDF Parsing**: Automatically reads and extracts clean tourist data from Wikipedia, blogs, and the provided local document `Odisha Tourism.pdf`.
2. **Robust Offline Bootstrapping**: Contains a pre-built knowledge base of major temples, beaches, food, and culture. If offline or if scraping fails, the database automatically populates itself.
3. **Advanced Anti-Hallucination RAG**: Prompts are strictly bounded to retrieved context, preventing the local LLM from generating false travel prices, schedules, or histories.
4. **Interactive Itinerary Builder**: Input travel hubs (Puri, Konark, Bhubaneswar), specify a duration (1-7 days), select a pace, and generate a day-by-day plan with meals and travel routes.
5. **One-Click PDF Export**: Instantly render and download a professional, beautifully styled PDF document of your generated travel plan.
6. **Voice Synthesis Tour Guide**: Integrated local text-to-speech engine (via `gTTS`) that speaks the guide's responses aloud.
7. **Premium Odisha-Branded UI**: Tailored interface using a custom warm terracotta and marigold HSL color palette, custom typography (`Outfit` and `Playfair Display` fonts), card animations, and side-by-side sources citation viewer.
8. **Multi-Language Support**: Seamlessly query and receive responses in **English, ଓଡ଼ିଆ (Odia), हिन्दी (Hindi), German (Deutsch), or French (Français)**.

---

## 🛠️ Step-by-Step Installation

### Prerequisites
*   **Python 3.11** installed on your system (highly recommended).
*   **Ollama** installed locally. Download it for free from [ollama.com](https://ollama.com/).

### 1. Set Up the Project Directory
Unpack the files or place them into your workspace:
```bash
cd c:\Users\hp\OneDrive\Desktop\my_rag
```

### 2. Create and Activate a Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup and Run Ollama
Open a separate terminal window and launch the Ollama server:
```bash
# Start the local Ollama background server
ollama serve
```

In a new terminal window, download the lightweight, high-performance local LLM model (`phi3` is recommended for standard CPUs/GPUs, or `llama3` for high-end GPUs):
```bash
# Pull the model
ollama pull phi3
```

---

## 🚀 Running the Application

Once the virtual environment is active, dependencies are installed, and Ollama is serving, launch the Flask web application server:

```bash
python app_flask.py
```

The server will initialize models and start listening on port 5000. Open your default web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📁 Modular Project Structure

```
my_rag/
│
├── app_flask.py              # Flask backend server exposing REST API routes
├── templates/                # Folder containing webpage templates
│   └── index.html            # Premium styled HTML5 single-page chat & itinerary dashboard
│
├── config.py                 # App parameters, paths, colors, and scraping URLs
├── requirements.txt          # Standard python dependencies (added flask package)
├── README.md                 # Complete manual & guide
├── ollama_setup.py          # Automatic local LLM check & download utility
├── scraper.py               # Live BeautifulSoup scraping + PDF reader + bootstrap facts
├── embedding.py             # Local HuggingFace embedding engine configuration
├── vector_store.py          # Local ChromaDB creation, splitters, and queries
├── rag_pipeline.py          # Anti-hallucination prompt, retrieval matching, and itineraries
├── app.py                   # High-fidelity custom Streamlit prototype (deprecating)
│
├── data/                    # Generated raw data directory (contains synthesized voice MP3s)
│   ├── raw/
│   └── processed/
│
├── vectordb/                # Persistent local database files folder
└── utils/                   # Shared travel helpers
    ├── __init__.py          # Exporter functions
    └── pdf_generator.py     # Tailored fpdf2 PDF itinerary converter
```

---

## 🔍 Troubleshooting Guide

### 1. "Ollama server is not running or not accessible"
*   **Reason**: The Ollama background service is not running.
*   **Fix**: Click the Ollama icon in your Windows system tray, or open a Command Prompt and run `ollama serve`. Make sure `http://localhost:11434` is accessible in your web browser.

### 2. Extremely Slow LLM Generation Times
*   **Reason**: Your system is running the model on the CPU instead of a dedicated GPU.
*   **Fix**: Ensure your computer has CUDA drivers installed if using an NVIDIA card. Alternatively, use a lighter model like `phi3` or `gemma:2b` by pulling them via `ollama pull gemma:2b` and selecting it in the Streamlit sidebar.

### 3. HuggingFace Connection Errors / Failures on First Run
*   **Reason**: The system is downloading the embedding model (`BAAI/bge-small-en-v1.5`) and has interrupted connectivity.
*   **Fix**: Verify your internet connection. On subsequent runs, the system loads the cached weights from `models/` completely offline. If `bge-small` fails due to local storage limits, the code automatically falls back to the lightweight `all-MiniLM-L6-v2` model.

### 4. Special Characters / Text Encoding Errors in PDFs
*   **Reason**: PDF generation fonts (like Helvetica) do not support deep multi-language glyphs (e.g. Odia characters).
*   **Fix**: The PDF generator implements a custom Latin-1 character converter that sanitizes characters. For deep regional script generation, export itineraries in English to ensure high-fidelity printing formats.
