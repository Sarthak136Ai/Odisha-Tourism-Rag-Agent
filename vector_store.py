import os
import glob
import logging
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import config
from embedding import get_embeddings_model

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class OdishaVectorStore:
    def __init__(self):
        self.embeddings = get_embeddings_model()
        
        # Check if we are running in Vercel or general cloud serverless environment
        is_cloud = os.environ.get("VERCEL") or os.environ.get("RENDER") or os.environ.get("PORT")
        if is_cloud:
            import shutil
            tmp_db_dir = "/tmp/vectordb"
            logger.info(f"Cloud deployment detected. Copying vector database from {config.VECTOR_DB_DIR} to {tmp_db_dir}...")
            try:
                # Copy existing pre-built DB to writable /tmp directory
                if os.path.exists(config.VECTOR_DB_DIR):
                    shutil.copytree(config.VECTOR_DB_DIR, tmp_db_dir, dirs_exist_ok=True)
                    logger.info("Successfully copied vector database to /tmp/vectordb.")
                else:
                    logger.warning(f"Source vector database directory {config.VECTOR_DB_DIR} does not exist!")
            except Exception as e:
                logger.error(f"Failed to copy vector database to /tmp: {e}")
            self.persist_directory = tmp_db_dir
        else:
            self.persist_directory = config.VECTOR_DB_DIR

        self.db = None
        self.initialize_db()

    def initialize_db(self):
        """
        Initializes the local Chroma database.
        """
        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="odisha_tourism"
            )
            logger.info(f"Initialized Chroma DB at: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Error initializing Chroma DB: {e}")
            raise e

    def _generate_chunk_id(self, content: str, source: str) -> str:
        """
        Generates a unique ID for a chunk based on a hash of its content and source.
        This prevents duplicate entries in the database.
        """
        hash_input = f"{content}_{source}".encode("utf-8")
        return hashlib.md5(hash_input).hexdigest()

    def process_raw_documents(self) -> list[Document]:
        """
        Reads raw text files in data/raw/, splits them into chunks, and wraps them in Documents.
        """
        raw_files = glob.glob(os.path.join(config.RAW_DATA_DIR, "*.txt"))
        if not raw_files:
            logger.warning("No raw text documents found to process in data/raw/")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

        all_chunks = []
        for file_path in raw_files:
            filename = os.path.basename(file_path)
            logger.info(f"Processing raw file: {filename}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Metadata source naming
            source_name = "Odisha Tourism Guide"
            if "wiki" in filename:
                source_name = f"Wikipedia ({filename.replace('scraped_', '').replace('.txt', '').replace('_', ' ').title()})"
            elif "unesco" in filename:
                source_name = "UNESCO World Heritage Page"
            elif "pdf" in filename:
                source_name = "Local Document (Odisha Tourism.pdf)"
            elif "bootstrap" in filename:
                source_name = "Curated Local Knowledge Base"

            chunks = text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                # We skip extremely small chunks (e.g. noise)
                if len(chunk.strip()) < 30:
                    continue
                
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": source_name,
                        "filename": filename,
                        "chunk_index": i,
                        "hash_id": self._generate_chunk_id(chunk, source_name)
                    }
                )
                all_chunks.append(doc)

        logger.info(f"Total processed chunks created: {len(all_chunks)}")
        return all_chunks

    def build_database(self, force_rebuild=False):
        """
        Processes documents and adds them to Chroma, ensuring no duplicate hash IDs are inserted.
        """
        chunks = self.process_raw_documents()
        if not chunks:
            logger.warning("No chunks available to index.")
            return

        # Check existing items in database
        existing_hashes = set()
        
        # If not forcing rebuild, we pull existing hashes to prevent duplicates
        if not force_rebuild:
            try:
                existing_data = self.db.get()
                if existing_data and "metadatas" in existing_data:
                    for meta in existing_data["metadatas"]:
                        if "hash_id" in meta:
                            existing_hashes.add(meta["hash_id"])
                logger.info(f"Loaded {len(existing_hashes)} existing document hashes from DB.")
            except Exception as e:
                logger.warning(f"Could not load existing metadata hashes: {e}")

        new_chunks = []
        new_ids = []
        
        for chunk in chunks:
            hid = chunk.metadata["hash_id"]
            if hid not in existing_hashes:
                new_chunks.append(chunk)
                new_ids.append(hid)
                # Keep track to avoid adding duplicates within the same batch
                existing_hashes.add(hid)

        if new_chunks:
            logger.info(f"Indexing {len(new_chunks)} new chunks into Chroma...")
            self.db.add_documents(documents=new_chunks, ids=new_ids)
            logger.info("Indexing complete! Database updated successfully.")
        else:
            logger.info("No new chunks to index. Vector Database is already up to date.")

    def get_retriever(self, k=4):
        """
        Returns the Chroma DB retriever object.
        """
        if self.db is None:
            self.initialize_db()
        return self.db.as_retriever(search_kwargs={"k": k})

    def query_similarity(self, query: str, k=4) -> list[dict]:
        """
        Queries the database directly and returns a list of matching chunks with source metadata.
        """
        if self.db is None:
            self.initialize_db()
        
        docs = self.db.similarity_search(query, k=k)
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "filename": doc.metadata.get("filename", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            })
        return results

if __name__ == "__main__":
    # Test DB creation
    # Make sure some raw data exists
    from scraper import OdishaTourismScraper
    scraper = OdishaTourismScraper()
    scraper.run_all_acquisition()
    
    logger.info("Initializing OdishaVectorStore...")
    ovs = OdishaVectorStore()
    logger.info("Building database...")
    ovs.build_database(force_rebuild=True)
    
    logger.info("Testing similarity query...")
    matches = ovs.query_similarity("How to travel to Puri and Jagannath Temple?", k=2)
    for i, m in enumerate(matches):
        print(f"\nMatch {i+1} [Source: {m['source']}]:")
        print(m['content'][:300] + "...")
