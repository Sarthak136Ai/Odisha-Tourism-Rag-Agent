import os
import shutil
import logging
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    model_name = "BAAI/bge-small-en-v1.5"
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bge-small-en-v1.5")
    
    logger.info(f"Downloading model '{model_name}' to local path: '{local_dir}'...")
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # Download the complete repository snapshot
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # ignore unnecessary formats to save bandwidth
        )
        logger.info("Successfully downloaded all model weights locally!")
        
        # Verify files
        files = os.listdir(local_dir)
        logger.info(f"Verified files in local model directory: {files}")
        print("\n[SUCCESS] Model downloaded successfully! You can now load it completely locally.")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        print("\n[ERROR] Download failed. Ensure you have an active internet connection.")

if __name__ == "__main__":
    main()
