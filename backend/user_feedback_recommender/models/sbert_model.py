import os
from sentence_transformers import SentenceTransformer, util
from logging_config import LoggerConfig

# Initialize logger
logger = LoggerConfig.get_logger("sbert")

# Set Hugging Face cache folder
HF_CACHE_DIR_MINI_LM_L6_V2 = "/home/user/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9"

HF_CACHE_DIR_MPNET_BASE_V2 = "/home/user/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/9a3225965996d404b775526de6dbfe85d3368642"

# HF_CACHE_DIR = "/home/user/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9"
os.environ["HF_HOME"] = HF_CACHE_DIR_MPNET_BASE_V2

class SBERTModel:
    _instance = None

    def __init__(self):
        """Private constructor to load SBERT model once."""
        logger.debug("Loading SBERT model from cache...")
        self.model = SentenceTransformer(HF_CACHE_DIR_MPNET_BASE_V2)
        logger.debug("SBERT Model successfully loaded!")
    
    @classmethod
    def get_model(cls):
        """Loads SBERTModel only once and returns the instance (not just the model)."""
        if cls._instance is None:
            logger.debug("Loading SBERT model from get_model...")
            cls._instance = cls()  # Store the SBERTModel instance
        return cls._instance
    

    def encode(self, texts, convert_to_tensor=True):
        """Encodes a list of texts using SBERT."""
        return self.model.encode(texts, convert_to_tensor=convert_to_tensor)
    

    def compute_similarity(cls, query_embedding, restaurant_embeddings):
        """Computes cosine similarity between query and restaurant embeddings."""
        return util.pytorch_cos_sim(query_embedding, restaurant_embeddings)