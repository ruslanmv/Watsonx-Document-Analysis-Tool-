#!/usr/bin/env python3
"""
Enhanced document ingestion script with product-specific collection support.

Usage:
    python scripts/ingest_docs.py --reset                    # Reset all collections  
    python scripts/ingest_docs.py --product dataflow_nexus   # Ingest specific product
    python scripts/ingest_docs.py --product cognos_bi        # Ingest specific product
    python scripts/ingest_docs.py                            # Ingest all products
"""

import os
import sys
import argparse
import logging
import time
import gc
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import (
    TextLoader, UnstructuredMarkdownLoader, 
    PyPDFLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai import APIClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHROMA_DB_PATH = "chroma_db"
DOCUMENTS_DIR = "documents"

# Product-specific collection mapping
PRODUCT_COLLECTIONS = {
    "dataflow_nexus": "dataflow_nexus_docs",
    "cognos_bi": "cognos_bi_docs", 
    "mixed": "documents"  # Default mixed collection
}

def get_embeddings():
    """Initialize Watson embeddings"""
    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_ENDPOINT = os.getenv("WATSONX_ENDPOINT", "https://us-south.ml.cloud.ibm.com")
    
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        raise ValueError("WATSONX_API_KEY and WATSONX_PROJECT_ID must be set in environment variables")
    
    credentials = {
        "url": WATSONX_ENDPOINT,
        "apikey": WATSONX_API_KEY
    }
    
    client = APIClient(
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID
    )
    
    return WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url=WATSONX_ENDPOINT,
        project_id=WATSONX_PROJECT_ID,
        params={"truncate_input_tokens": 512},
        watsonx_client=client
    )

def load_documents_for_product(product_name: str) -> List:
    """Load documents for a specific product"""
    product_dir = os.path.join(DOCUMENTS_DIR, product_name)
    
    if not os.path.exists(product_dir):
        logger.warning(f"Product directory {product_dir} does not exist")
        return []
    
    logger.info(f"Loading documents from {product_dir}...")
    
    documents = []
    
    # Load all files in the product directory
    for root, dirs, files in os.walk(product_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, ".")
            
            try:
                if file.endswith('.md'):
                    loader = UnstructuredMarkdownLoader(file_path)
                elif file.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                logger.info(f"Loading {relative_path}...")
                docs = loader.load()
                
                # Add product metadata to each document
                for doc in docs:
                    doc.metadata.update({
                        'product': product_name,
                        'category': product_name,
                        'source': relative_path
                    })
                
                documents.extend(docs)
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents for {product_name}")
    return documents

def chunk_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """Split documents into chunks"""
    if not documents:
        return []
        
    logger.info(f"Chunking {len(documents)} documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk info and count tokens
    for i, chunk in enumerate(chunks):
        # Simple token counting (approximate)
        token_count = len(chunk.page_content.split())
        chunk.metadata['chunk_id'] = i
        chunk.metadata['token_count'] = token_count
        logger.info(f"Chunk {i}: {token_count} tokens")
    
    logger.info(f"Generated {len(chunks)} chunks.")
    return chunks

def reset_database(chroma_path: str = CHROMA_DB_PATH, max_retries: int = 3):
    """Reset ChromaDB with retry logic for Windows"""
    logger.info("🔨 Resetting ChromaDB database...")
    
    for attempt in range(max_retries):
        try:
            if os.path.exists(chroma_path):
                # Force garbage collection
                gc.collect()
                time.sleep(1)
                
                # Try to remove the directory
                shutil.rmtree(chroma_path)
                logger.info(f"✅ Database reset successful (attempt {attempt + 1})")
                return
                
        except PermissionError as e:
            logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"🔄 Retrying in 2 seconds...")
                time.sleep(2)
                gc.collect()
            else:
                logger.error("❌ Database reset failed after all retries")
                logger.error("💡 Try running: python scripts/force_reset_db.py")
                raise
        except Exception as e:
            logger.error(f"❌ Unexpected error during reset: {e}")
            raise

def ingest_product_documents(product_name: str, reset: bool = False):
    """Ingest documents for a specific product"""
    collection_name = PRODUCT_COLLECTIONS.get(product_name, f"{product_name}_docs")
    
    logger.info(f"🚀 Starting {product_name} document ingestion...")
    logger.info(f"   Collection name: {collection_name}")
    logger.info(f"   Database path: {CHROMA_DB_PATH}")
    logger.info(f"   Reset database: {reset}")
    
    if reset:
        reset_database()
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get or create collection
    try:
        collection = client.get_collection(collection_name)
        if reset:
            client.delete_collection(collection_name)
            collection = client.create_collection(collection_name)
        else:
            logger.info(f"Using existing collection: {collection_name}")
    except:
        collection = client.create_collection(collection_name)
    
    # Load and process documents
    documents = load_documents_for_product(product_name)
    if not documents:
        logger.warning(f"No documents found for product: {product_name}")
        return
    
    chunks = chunk_documents(documents)
    if not chunks:
        logger.warning(f"No chunks generated for product: {product_name}")
        return
    
    # Get embeddings
    embeddings = get_embeddings()
    
    # Add to ChromaDB
    logger.info(f"Adding {len(chunks)} chunks to collection '{collection_name}'...")
    
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"{product_name}_{i}" for i in range(len(chunks))]
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embedding_vectors = embeddings.embed_documents(texts)
    
    # Add to collection
    collection.add(
        embeddings=embedding_vectors,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info(f"✅ Documents successfully ingested into ChromaDB collection: '{collection_name}'")
    logger.info(f"📊 Total documents in collection '{collection_name}': {collection.count()}")

def ingest_all_products(reset: bool = False):
    """Ingest documents for all products"""
    if reset:
        reset_database()
    
    # Find all product directories
    products = []
    if os.path.exists(DOCUMENTS_DIR):
        for item in os.listdir(DOCUMENTS_DIR):
            item_path = os.path.join(DOCUMENTS_DIR, item)
            if os.path.isdir(item_path):
                products.append(item)
    
    if not products:
        logger.warning(f"No product directories found in {DOCUMENTS_DIR}")
        return
    
    logger.info(f"Found products: {products}")
    
    for product in products:
        try:
            ingest_product_documents(product, reset=False)  # Don't reset for each product
        except Exception as e:
            logger.error(f"Failed to ingest {product}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB with product-specific collections")
    parser.add_argument("--reset", action="store_true", help="Reset the database before ingesting")
    parser.add_argument("--product", type=str, help="Ingest documents for specific product only")
    parser.add_argument("--list-products", action="store_true", help="List available products")
    
    args = parser.parse_args()
    
    if args.list_products:
        if os.path.exists(DOCUMENTS_DIR):
            products = [item for item in os.listdir(DOCUMENTS_DIR) 
                       if os.path.isdir(os.path.join(DOCUMENTS_DIR, item))]
            print("Available products:")
            for product in products:
                collection_name = PRODUCT_COLLECTIONS.get(product, f"{product}_docs")
                print(f"  - {product} (collection: {collection_name})")
        else:
            print(f"Documents directory {DOCUMENTS_DIR} not found")
        return
    
    try:
        if args.product:
            ingest_product_documents(args.product, args.reset)
        else:
            ingest_all_products(args.reset)
    except KeyboardInterrupt:
        logger.info("❌ Ingestion cancelled by user")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 