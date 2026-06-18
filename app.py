import os
import time
import logging
import streamlit as st
from gtts import gTTS
import config
from scraper import OdishaTourismScraper, BOOTSTRAP_DATA
from vector_store import OdishaVectorStore
from rag_pipeline import OdishaRAGPipeline
from ollama_setup import OllamaOrchestrator
from utils.pdf_generator import generate_itinerary_pdf

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- STAGE 1: INJECTING PREMIUM STYLING ---
st.set_page_config(
    page_title="Kalinga GPT",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load modern Outfit and Playfair Display fonts and inject custom styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

    /* Variables and Theme colors */
    :root {
        --primary: #D4AF37;
        --secondary: #E07A5F;
        --bg: #F4F1DE;
        --text: #3D405B;
        --accent: #81B29A;
    }

    /* Core typography overrides */
    .stApp {
        background-color: #F8F7F2;
        color: #3D405B;
        font-family: 'Outfit', sans-serif;
    }

    /* Banner Container */
    .banner-container {
        background: linear-gradient(135deg, #E07A5F 0%, #D4AF37 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(224, 122, 95, 0.15);
    }
    .banner-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.15);
    }
    .banner-subtitle {
        font-size: 1.15rem;
        font-weight: 300;
        opacity: 0.95;
    }

    /* Chat bubble custom aesthetics */
    .chat-bubble {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        line-height: 1.5;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
    }
    .user-bubble {
        background-color: #FFFFFF;
        border-left: 5px solid #81B29A;
        text-align: left;
    }
    .assistant-bubble {
        background-color: #F4F1DE;
        border-left: 5px solid #E07A5F;
        text-align: left;
    }
    .bubble-header {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
        letter-spacing: 0.5px;
    }
    .user-hdr { color: #81B29A; }
    .assistant-hdr { color: #E07A5F; }

    /* Custom Cards */
    .info-card {
        background-color: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
        border-top: 4px solid #D4AF37;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.15);
    }

    /* Quick Suggestions Buttons */
    .stButton>button {
        background-color: #FFFFFF;
        color: #3D405B;
        border: 1px solid #E07A5F;
        border-radius: 20px;
        padding: 0.4rem 1.2rem;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    .stButton>button:hover {
        background-color: #E07A5F;
        color: white;
        border-color: #E07A5F;
        box-shadow: 0 4px 12px rgba(224, 122, 95, 0.2);
    }
    
    /* Active Model Indicator style */
    .active-badge {
        background-color: #81B29A;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
    }

    /* Custom style for Streamlit Tabs text */
    .stTabs [data-baseweb="tab"] p {
        color: #3D405B !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        transition: color 0.3s ease !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #E07A5F !important;
    }

    /* Custom themed scrollbars strictly for the chat container */
    .chat-scroll-container {
        height: 450px !important;
        overflow-y: scroll !important;
        scrollbar-width: auto !important;
        scrollbar-color: #E07A5F #F4F1DE !important;
        padding-right: 15px !important;
        margin-right: 5px !important;
        border-right: 1px solid rgba(224, 122, 95, 0.15) !important;
    }
    .chat-scroll-container::-webkit-scrollbar {
        width: 12px !important;
        height: 12px !important;
        display: block !important;
    }
    .chat-scroll-container::-webkit-scrollbar-track {
        background: #F4F1DE !important;
        border-radius: 6px !important;
        box-shadow: inset 0 0 6px rgba(0,0,0,0.05) !important;
    }
    .chat-scroll-container::-webkit-scrollbar-thumb {
        background-color: #E07A5F !important;
        border-radius: 6px !important;
        border: 2px solid #F4F1DE !important; /* Premium floating look */
    }
    .chat-scroll-container::-webkit-scrollbar-thumb:hover {
        background-color: #D4AF37 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- STAGE 2: INITIALIZING CORE PIPELINES ---
@st.cache_resource
def get_pipelines():
    """
    Caches the instantiation of vector store and rag pipeline for speedy loads.
    """
    scraper_inst = OdishaTourismScraper()
    # Check if vector DB needs initial building
    vector_inst = OdishaVectorStore()
    
    # Try bootstrapping if DB is empty
    try:
        existing = vector_inst.db.get()
        if not existing or len(existing.get("documents", [])) == 0:
            logger.info("Vector database is empty. Running initial bootstrapping...")
            scraper_inst.run_all_acquisition()
            vector_inst.build_database()
    except Exception as e:
        logger.warning(f"Error checking DB size, running bootstrap: {e}")
        scraper_inst.run_all_acquisition()
        vector_inst.build_database()
        
    rag_inst = OdishaRAGPipeline()
    ollama_inst = OllamaOrchestrator()
    return scraper_inst, vector_inst, rag_inst, ollama_inst

try:
    scraper, vector_store, rag_pipeline, ollama_orchestrator = get_pipelines()
except Exception as e:
    st.error(f"Critical System Loading Error: {e}")
    st.info("Ensure PyTorch and other standard frameworks are successfully installed in your Python environment.")
    st.stop()


# --- STAGE 3: SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("### 🕌 Kalinga GPT PORTAL")
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/1a/Seal_of_Odisha.svg", width=100)
    
    st.markdown("---")
    st.markdown("#### ⚙️ LOCAL LLM CONFIGURATION")
    
    # Language Selector
    lang_code = st.selectbox(
        "Select Portal Language:",
        options=list(config.LANGUAGES.keys()),
        format_func=lambda x: config.LANGUAGES[x]
    )
    
    # Model Selector
    installed_models = ollama_orchestrator.list_local_models()
    selected_model = st.selectbox(
        "Active Local LLM:",
        options=installed_models if installed_models else [config.DEFAULT_LLM_MODEL],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### 🩺 SYSTEM DIAGNOSTICS")
    
    # Check status values
    ollama_active = ollama_orchestrator.check_ollama_status()
    db_document_count = 0
    try:
        db_data = vector_store.db.get()
        if db_data and "documents" in db_data:
            db_document_count = len(db_data["documents"])
    except Exception:
        pass
        
    status_color = "🟢 ONLINE" if ollama_active else "🔴 OFFLINE"
    st.markdown(f"**Local Ollama Server:** {status_color}")
    st.markdown(f"**Active Model:** `{selected_model}`")
    st.markdown(f"**Vector Database:** `ChromaDB` ({db_document_count} chunks)")
    
    st.markdown("---")
    st.markdown("#### 🔄 DATABASE UPDATE CENTER")
    st.markdown('<div style="background-color: #FFF3CD; border-left: 5px solid #FFC107; padding: 0.75rem; border-radius: 8px; color: #856404; font-size: 0.9rem; font-weight: 500; margin-bottom: 1rem;">Perform a real-time web scrape and PDF scan to update the local vector knowledge base.</div>', unsafe_allow_html=True)
    
    if st.button("Index & Rebuild Vector DB"):
        with st.spinner("Acquiring data and rebuilding vector indexes..."):
            try:
                results = scraper.run_all_acquisition()
                vector_store.build_database(force_rebuild=True)
                st.success("Database successfully updated!")
                time.sleep(1.5)
                st.rerun()
            except Exception as ex:
                st.error(f"Error during re-indexing: {ex}")

    st.markdown("""
    <div style='text-align: center; margin-top: 2rem; color: #888; font-size: 0.8rem;'>
        Odisha RAG Agent v1.0<br/>Fully Local & Offline AI
    </div>
    """, unsafe_allow_html=True)


# --- STAGE 4: MAIN HEADER BANNER ---
st.markdown("""
<div class="banner-container">
    <div class="banner-title">Kalinga GPT</div>
    <div class="banner-subtitle">Explore "India's Best Kept Secret" powered by fully secure, local generative AI</div>
</div>
""", unsafe_allow_html=True)


# --- STAGE 5: SEPARATING USER INTENTS (TABS) ---
tab_chat, tab_itinerary, tab_heritage = st.tabs([
    "💬 Live AI Chatbot", 
    "🗺️ Interactive Itinerary Planner", 
    "📜 Heritage & Travel Wiki"
])

# Initialize session chat state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==============================================================================
# TAB 1: LIVE CHATBOT INTERFACE
# ==============================================================================
with tab_chat:
    # 1. Quick suggestion chips
    st.markdown("##### 💡 Ask about:")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    suggested_q = None
    
    with col_s1:
        if st.button("Puri Temple Entry Rules 🕌"):
            suggested_q = "What are the rules, timings, and dress codes for entering the Puri Jagannath Temple?"
    with col_s2:
        if st.button("Konark Wheel Mechanism ☀️"):
            suggested_q = "How do the carved wheels of Konark Sun Temple work as sundials?"
    with col_s3:
        if st.button("Chilika Lake Dolphins 🐬"):
            suggested_q = "Where and when can I spot Irrawaddy Dolphins and migratory birds in Chilika?"
    with col_s4:
        if st.button("Traditional Odia Cuisines 🍲"):
            suggested_q = "What are the local cuisines and signature sweets of Odisha I must try?"
    # Chat layouts split: Left = Chat Window, Right = Interactive Citations
    chat_col, citation_col = st.columns([5, 3])
    
    with chat_col:
        st.markdown("#### 🗣️ Conversational Guide")
        
        # Trigger query variable
        triggered_query = None
        
        # 1. Capture suggested chip inputs if clicked
        if suggested_q:
            triggered_query = suggested_q
            
        # Helper to define the input form
        def render_chat_input_form(form_key):
            with st.form(key=form_key, clear_on_submit=True):
                input_col, button_col = st.columns([5, 1])
                with input_col:
                    u_query = st.text_input("Enter your travel query...", placeholder="Ask about temples, cuisines, dolphins, rules...", label_visibility="collapsed")
                with button_col:
                    submit_q = st.form_submit_button("Send 📤")
            if submit_q and u_query.strip():
                return u_query.strip()
            return None

        # --- CONDITION 1: CHAT IS EMPTY (Input at the top!) ---
        if not st.session_state.chat_history:
            st.markdown('<p style="color:#777; font-size:0.9rem; margin-bottom:0.5rem;">Start your travel planning by asking your first question below:</p>', unsafe_allow_html=True)
            form_query = render_chat_input_form("empty_chat_form")
            if form_query:
                triggered_query = form_query
                
        # --- CONDITION 2: CHAT HAS MESSAGES (Input goes down, contents go up!) ---
        else:
            # Display chat history inside a custom HTML scroll container
            chat_html = ""
            for message in st.session_state.chat_history:
                role_class = "user-bubble" if message["role"] == "user" else "assistant-bubble"
                role_hdr = "User" if message["role"] == "user" else "Ollama Guide"
                hdr_class = "user-hdr" if message["role"] == "user" else "assistant-hdr"
                
                # Concatenate as a clean flat string without leading newlines or spacing to prevent code block parsing
                chat_html += f'<div class="chat-bubble {role_class}"><div class="bubble-header {hdr_class}">{role_hdr}</div><div>{message["content"]}</div></div>'
            
            # Render the flat HTML wrapper without indentations
            st.markdown(f'<div class="chat-scroll-container">{chat_html}</div>', unsafe_allow_html=True)
            
            # Text input field goes below the messages
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            form_query = render_chat_input_form("active_chat_form")
            if form_query:
                triggered_query = form_query

        # Handle RAG execution
        if triggered_query:
            # 1. Append user question
            st.session_state.chat_history.append({"role": "user", "content": triggered_query})
            
            # Show processing placeholder
            with st.spinner("Searching database and thinking..."):
                start_time = time.time()
                # Run RAG
                rag_response = rag_pipeline.run_pipeline(
                    query=triggered_query,
                    model_name=selected_model,
                    language=lang_code
                )
                duration = time.time() - start_time
                
            # 2. Append assistant response
            answer = rag_response["answer"]
            citations = rag_response["citations"]
            
            # Add generation metadata
            formatted_answer = f"{answer}\n\n*⏱️ Generated locally in {duration:.2f} seconds.*"
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": formatted_answer,
                "citations": citations
            })
            st.rerun()

    with citation_col:
        st.markdown("#### 📑 Verified Sources & Citations")
        st.markdown('<p style="color: #3D405B; font-size: 0.9rem; opacity: 0.9; font-weight: 500; margin-bottom: 1.25rem;">Context chunks retrieved in real-time from the local vector database used to formulate the guide\'s answer.</p>', unsafe_allow_html=True)
        
        # Get the citations of the last assistant message
        last_citations = []
        if st.session_state.chat_history:
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "assistant" and "citations" in msg:
                    last_citations = msg["citations"]
                    break
                    
        if last_citations:
            for idx, cit in enumerate(last_citations):
                st.markdown(f"""
                <div class="info-card">
                    <strong style="color: #E07A5F;">Citation {idx+1}: {cit['source']}</strong><br/>
                    <small style="color:#777;">File: {cit['filename']} | Chunk ID: {cit['chunk_index']}</small>
                    <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #eee;"/>
                    <p style="font-size: 0.9rem; line-height: 1.4; color: #3D405B;">"{cit['content']}"</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="background-color: #E8F0FE; border-left: 5px solid #1A73E8; padding: 1rem; border-radius: 8px; color: #1967D2; font-weight: 600; font-size: 0.95rem; line-height: 1.5;">Ask a question to see real-time vector citations and source page tracking.</div>', unsafe_allow_html=True)

        # --- VOICE ASSISTANT COMPONENT ---
        st.markdown("---")
        st.markdown("#### 🔊 Audio Tour Guide")
        
        last_answer = ""
        if st.session_state.chat_history:
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "assistant":
                    # Remove processing metadata for clean TTS reading
                    last_answer = msg["content"].split("⏱️ Generated locally")[0].strip()
                    break
                    
        if last_answer:
            # TTS generation
            if st.button("Read Answer Aloud 🔈"):
                with st.spinner("Synthesizing local voice guide..."):
                    try:
                        tts_lang = "or" if lang_code == "or" else ("hi" if lang_code == "hi" else "en")
                        tts = gTTS(text=last_answer, lang=tts_lang, slow=False)
                        voice_path = os.path.join(config.DATA_DIR, "voice_response.mp3")
                        tts.save(voice_path)
                        st.audio(voice_path, format="audio/mp3")
                    except Exception as tts_err:
                        st.error(f"Text-To-Speech engine encountered an issue: {tts_err}")
        else:
            st.markdown('<p style="color: #3D405B; font-size: 0.85rem; opacity: 0.9; font-weight: 500;">Voice Guide is ready. Simply ask a question and click read.</p>', unsafe_allow_html=True)


# ==============================================================================
# TAB 2: INTERACTIVE ITINERARY PLANNER
# ==============================================================================
with tab_itinerary:
    st.markdown("#### 🗺️ Custom Travel Plan Generator")
    st.write("Generate a bespoke travel itinerary drawing directly from verified vector-database sources.")
    
    col_i1, col_i2, col_i3 = st.columns(3)
    
    with col_i1:
        destination = st.selectbox(
            "Target Destination Hub:",
            options=["Puri & Konark (Spiritual & Heritage)", "Bhubaneswar (Temple City & Caves)", "Chilika Lake & Nature Sanctuaries", "Custom Target"]
        )
        if destination == "Custom Target":
            custom_dest = st.text_input("Specify City / Attraction:", value="Sambalpur")
            target_destination = custom_dest
        else:
            target_destination = destination
            
    with col_i2:
        duration = st.slider("Duration (Days):", min_value=1, max_value=7, value=3)
        
    with col_i3:
        pace_travel = st.selectbox("Travel Pace:", options=["relaxed", "balanced", "intense"], index=1)
        
    if st.button("Generate Travel Itinerary 🚀"):
        with st.spinner("Mapping routes and culinary recommendations..."):
            itinerary_result = rag_pipeline.generate_custom_itinerary(
                location=target_destination,
                duration_days=duration,
                pace=pace_travel
            )
            # Store in session state to enable downloading
            st.session_state.active_itinerary = itinerary_result
            st.session_state.active_dest = target_destination
            st.session_state.active_dur = f"{duration} Days"
            
    if "active_itinerary" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state.active_itinerary)
        
        # Exporter Block
        st.markdown("---")
        st.markdown("##### 📥 Export Itinerary")
        
        pdf_filename = f"{st.session_state.active_dest.replace(' ', '_')}_itinerary.pdf"
        pdf_filepath = os.path.join(config.DATA_DIR, pdf_filename)
        
        if st.button("Convert to PDF Itinerary 📄"):
            with st.spinner("Creating beautiful PDF file..."):
                success = generate_itinerary_pdf(
                    itinerary_text=st.session_state.active_itinerary,
                    filename=pdf_filepath,
                    duration_days=st.session_state.active_dur,
                    traveler_name="Odisha Explorer"
                )
                if success:
                    st.success(f"PDF Successfully created at: {pdf_filepath}")
                    with open(pdf_filepath, "rb") as f:
                        st.download_button(
                            label="📥 Download PDF Document",
                            data=f,
                            file_name=pdf_filename,
                            mime="application/pdf"
                        )
                else:
                    st.error("Failed to generate PDF document.")


# ==============================================================================
# TAB 3: ODISHA HERITAGE & TRAVEL WIKI
# ==============================================================================
with tab_heritage:
    st.markdown("#### 📜 Heritage & Travel Encyclopedia")
    st.write("Browse pre-classified offline bootstrap knowledge and local document cards.")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    
    # Categorize bootstrap entries
    temples_list = [b for b in BOOTSTRAP_DATA if b["category"] == "Temples & Heritage"]
    wildlife_list = [b for b in BOOTSTRAP_DATA if b["category"] == "Wildlife & Nature"]
    food_culture = [b for b in BOOTSTRAP_DATA if b["category"] in ["Local Food", "Culture & Art", "Festivals"]]
    
    with col_c1:
        st.markdown("### 🕌 Temples & History")
        for entry in temples_list:
            with st.expander(entry["title"]):
                st.markdown(entry["content"])
                
    with col_c2:
        st.markdown("### 🐬 Nature & Wildlife")
        for entry in wildlife_list:
            with st.expander(entry["title"]):
                st.markdown(entry["content"])
                
    with col_c3:
        st.markdown("### 🍲 Culture & Food")
        for entry in food_culture:
            with st.expander(entry["title"]):
                st.markdown(entry["content"])
