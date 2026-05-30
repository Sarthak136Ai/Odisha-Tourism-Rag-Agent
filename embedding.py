import os
import logging
from typing import List
import requests
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class HuggingFaceInferenceEmbeddings:
    """
    Zero-RAM cloud embedding generator. 
    Queries Hugging Face's free public Inference API directly over HTTP.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        base_name = os.path.basename(model_name)
        if base_name == "bge-small-en-v1.5":
            self.model_id = "BAAI/bge-small-en-v1.5"
        else:
            self.model_id = model_name
        
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_id}"
        # Optional token for higher rate limits in production
        self.token = os.environ.get("HF_API_KEY", "")
        self.headers = {}
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def _embed(self, texts: List[str]) -> List[List[float]]:
        try:
            logger.info(f"Generating embeddings via HF Cloud Inference API for {len(texts)} chunks...")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=15
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"HF Inference API returned status {response.status_code}. Falling back to zero-vectors.")
                return [[0.0] * 384 for _ in texts]
        except Exception as e:
            logger.error(f"HF Inference API connection error: {e}")
            return [[0.0] * 384 for _ in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


def get_embeddings_model():
    """
    Initializes and returns the appropriate HuggingFace Embeddings model.
    Routes to zero-RAM Cloud API in production, and high-fidelity PyTorch offline model locally.
    """
    # Detect if we are running in the cloud
    if os.environ.get("RENDER") or os.environ.get("PORT") or os.environ.get("VERCEL"):
        logger.info("Cloud deployment detected! Activating ultra-lightweight HuggingFace Inference API embeddings (0MB RAM)...")
        return HuggingFaceInferenceEmbeddings()

    # Local PC execution - load heavy PyTorch modules lazily to preserve cloud RAM
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA is available! Using GPU for generating embeddings.")
    else:
        device = "cpu"
        logger.info("CUDA is not available. Using CPU for generating embeddings.")

    # Configure caching directories inside the project models directory
    os.environ["HF_HOME"] = config.MODELS_DIR
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = config.MODELS_DIR

    model_name = config.EMBEDDING_MODEL_NAME
    fallback_model = config.FALLBACK_EMBEDDING_MODEL
    
    # Embedding model arguments
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True} # Recommended for BGE embeddings to use cosine similarity

    try:
        logger.info(f"Loading local offline embedding model: {model_name}...")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info("Successfully loaded local embedding model!")
        return embeddings
    except Exception as e:
        logger.warning(f"Failed to load primary embedding model '{model_name}' due to: {e}")
        logger.info(f"Attempting to load fallback model '{fallback_model}'...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=fallback_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            logger.info("Successfully loaded fallback embedding model!")
            return embeddings
        except Exception as fe:
            logger.critical(f"Failed to load fallback embedding model: {fe}")
            logger.info("Please ensure you have an active internet connection on the first run to download sentence-transformers.")
            raise fe

if __name__ == "__main__":
    # Test local embeddings loading
    embeddings = get_embeddings_model()
    test_text = "Odisha is beautiful and famous for the Konark Sun Temple."
    vector = embeddings.embed_query(test_text)
    print(f"Embedding model tested successfully! Vector length: {len(vector)}")
