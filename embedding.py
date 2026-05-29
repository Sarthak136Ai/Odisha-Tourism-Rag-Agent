import os
import torch
import logging
from langchain_huggingface import HuggingFaceEmbeddings
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_embeddings_model():
    """
    Initializes and returns the local HuggingFace Embeddings model.
    It automatically detects GPU (CUDA) availability for faster inference.
    """
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
        logger.info(f"Loading local embedding model: {model_name}...")
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
