"""
Embeddings utility for LangChain: supports only IBM watsonx.

Usage:
    from agent.embeddings import get_embeddings
    embeddings = get_embeddings(model_id="ibm/slate-125m-english-rtrvr-v2")
"""
import os
from dotenv import load_dotenv

# Try to load .env from project root or current dir
load_dotenv()

# IBM watsonx imports
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import Embeddings as IBMEmbeddings
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

class IBMEmbeddingWrapper:
    def __init__(self, emb):
        self.emb = emb

    def embed_documents(self, texts):
        return self.emb.embed_documents(texts=texts)

    def embed_query(self, text):
        # Chroma expects this to return a list/array, not a batch
        return self.emb.embed_documents([text])[0]

def get_ibm_embeddings(model_id=None):
    if not IBM_AVAILABLE:
        raise ImportError("ibm_watsonx_ai not installed.")
    model_id = model_id or "ibm/slate-125m-english-rtrvr-v2"
    API_KEY = os.getenv("WATSONX_API_KEY")
    PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    ENDPOINT = os.getenv("WATSONX_ENDPOINT", "https://us-south.ml.cloud.ibm.com")
    credentials = Credentials(url=ENDPOINT, api_key=API_KEY)
    emb = IBMEmbeddings(model_id=model_id, credentials=credentials, project_id=PROJECT_ID)
    return IBMEmbeddingWrapper(emb)

# Alias for compatibility
get_embeddings = get_ibm_embeddings 