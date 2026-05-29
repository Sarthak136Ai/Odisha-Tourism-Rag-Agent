import json
import logging
import requests
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class OllamaOrchestrator:
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL

    def check_ollama_status(self) -> bool:
        """
        Pings the Ollama local server to see if it is active.
        """
        try:
            logger.info(f"Pinging Ollama local server at: {self.base_url}...")
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                logger.info("Ollama server is active and responding!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        logger.error("Ollama server is not running or not accessible.")
        logger.error("Please make sure you have downloaded Ollama (https://ollama.com) and it is actively running in the background.")
        logger.error("To start the server manually in command prompt or powershell: 'ollama serve'")
        return False

    def list_local_models(self) -> list[str]:
        """
        Queries the Ollama server for currently installed models.
        """
        if not self.check_ollama_status():
            return []
            
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model["name"].split(":")[0] for model in data.get("models", [])]
                logger.info(f"Locally installed Ollama models: {models}")
                return models
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
        return []

    def pull_model(self, model_name: str) -> bool:
        """
        Instructs the local Ollama server to download a model.
        Supports streaming logs of download progress.
        """
        if not self.check_ollama_status():
            return False

        logger.info(f"Requesting download (pull) of model: '{model_name}'. Please wait, this may take a few minutes depending on connection...")
        try:
            url = f"{self.base_url}/api/pull"
            payload = {"name": model_name, "stream": False}
            
            response = requests.post(url, json=payload, timeout=600) # Long timeout for model download
            if response.status_code == 200:
                logger.info(f"Successfully downloaded/pulled local LLM model: '{model_name}'!")
                return True
            else:
                logger.error(f"Failed to pull model. Server responded with status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Exception raised while pulling model: {e}")
            logger.info(f"Fallback suggestion: Please open a terminal and run 'ollama pull {model_name}' manually.")
            return False

    def ensure_model_exists(self, model_name: str) -> bool:
        """
        Ensures a model is available locally. If not, pulls it.
        """
        local_models = self.list_local_models()
        
        # Check standard name as well as versioned tag
        if model_name in local_models or any(m.startswith(model_name) for m in local_models):
            logger.info(f"Model '{model_name}' is already installed locally and ready for use!")
            return True
            
        logger.info(f"Model '{model_name}' was not found in local installations.")
        return self.pull_model(model_name)

if __name__ == "__main__":
    orchestrator = OllamaOrchestrator()
    if orchestrator.check_ollama_status():
        # Test pulling phi3
        orchestrator.ensure_model_exists(config.DEFAULT_LLM_MODEL)
    else:
        print("\n[WARNING] Ollama server is offline! Please start Ollama by running 'ollama serve' in your terminal.")
