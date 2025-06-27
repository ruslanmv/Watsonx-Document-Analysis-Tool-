"""
Configuration management for Requirements Analysis system
Enhanced with product-specific collection support
"""
import os
import chromadb
from typing import Dict, List, Tuple, Optional

# Base configuration
DEFAULT_COLLECTION = "documents"
DEFAULT_CHROMA_DB_PATH = "chroma_db"

# Product-specific collection mapping
PRODUCT_COLLECTIONS = {
    "dataflow_nexus": "dataflow_nexus_docs",
    "cognos_bi": "cognos_bi_docs",
    "test_product": "test_product_docs",  # Test collection with EnterpriseFlow Platform
    "mixed": "documents"  # Default mixed collection
}

def get_chroma_db_path() -> str:
    """Get ChromaDB path from environment or default"""
    return os.getenv('CHROMA_DB_PATH', DEFAULT_CHROMA_DB_PATH)

def get_collection_name(product: Optional[str] = None) -> str:
    """
    Get collection name for specific product or detect best available collection
    
    Args:
        product: Product name (e.g., 'dataflow_nexus', 'cognos_bi')
        
    Returns:
        Collection name to use
    """
    if product and product in PRODUCT_COLLECTIONS:
        return PRODUCT_COLLECTIONS[product]
    
    # Auto-detect best collection if no product specified
    return get_best_collection_name()

def get_best_collection_name() -> str:
    """
    Auto-detect the best collection to use based on:
    1. Environment variable COLLECTION_NAME
    2. Prioritize test_product_docs if available (for testing)
    3. Available collections (prioritize those with most documents)
    """
    # Check environment variable first
    env_collection = os.getenv('COLLECTION_NAME')
    if env_collection:
        return env_collection
    
    # Prioritize test_product_docs for testing/demo purposes
    try:
        available_collections = list_available_collections()
        collection_names = [name for name, count in available_collections]
        
        # Check if test_product_docs exists and has data
        if "test_product_docs" in collection_names:
            for name, count in available_collections:
                if name == "test_product_docs" and count > 0:
                    print(f"🧪 Using test collection 'test_product_docs' with {count} documents (EnterpriseFlow Platform)")
                    return "test_product_docs"
        
        # Fall back to collection with most documents
        if available_collections:
            # Sort by document count (descending)
            available_collections.sort(key=lambda x: x[1], reverse=True)
            best_collection = available_collections[0][0]
            print(f"Auto-detected collection '{best_collection}' with {available_collections[0][1]} documents")
            return best_collection
            
    except Exception as e:
        print(f"Warning: Could not auto-detect collection: {e}")
    
    # Fallback to default
    return DEFAULT_COLLECTION

def list_available_collections() -> List[Tuple[str, int]]:
    """
    List all available ChromaDB collections with document counts
    
    Returns:
        List of (collection_name, document_count) tuples
    """
    try:
        client = chromadb.PersistentClient(path=get_chroma_db_path())
        collections = client.list_collections()
        
        result = []
        for collection_info in collections:
            collection = client.get_collection(collection_info.name)
            count = collection.count()
            result.append((collection_info.name, count))
        
        return result
    except Exception as e:
        print(f"Warning: Could not list collections: {e}")
        return []

def get_available_products() -> Dict[str, str]:
    """
    Get available products based on existing collections
    
    Returns:
        Dict mapping product display names to collection names
    """
    products = {}
    collections = list_available_collections()
    
    for collection_name, count in collections:
        if count > 0:  # Only include collections with documents
            # Map collection names back to products
            for product, mapped_collection in PRODUCT_COLLECTIONS.items():
                if collection_name == mapped_collection:
                    display_name = product.replace('_', ' ').title()
                    products[display_name] = {
                        'collection': collection_name,
                        'product_key': product,
                        'document_count': count
                    }
                    break
            else:
                # Handle unknown collections
                if collection_name not in [v for v in PRODUCT_COLLECTIONS.values()]:
                    display_name = collection_name.replace('_', ' ').title()
                    products[display_name] = {
                        'collection': collection_name,
                        'product_key': collection_name,
                        'document_count': count
                    }
    
    return products

def get_product_collection_metadata(product_key: str) -> Dict:
    """
    Get metadata for a specific product's collection
    
    Args:
        product_key: Product key (e.g., 'dataflow_nexus')
        
    Returns:
        Dict with collection metadata
    """
    collection_name = get_collection_name(product_key)
    
    try:
        client = chromadb.PersistentClient(path=get_chroma_db_path())
        collection = client.get_collection(collection_name)
        count = collection.count()
        
        # Get sample to understand content
        if count > 0:
            sample = collection.peek(limit=1)
            metadata = sample['metadatas'][0] if sample['metadatas'] else {}
        else:
            metadata = {}
        
        return {
            'collection_name': collection_name,
            'document_count': count,
            'product': metadata.get('product', product_key),
            'category': metadata.get('category', 'unknown'),
            'available': count > 0
        }
    except Exception as e:
        return {
            'collection_name': collection_name,
            'document_count': 0,
            'product': product_key,
            'category': 'unknown',
            'available': False,
            'error': str(e)
        }

if __name__ == "__main__":
    print("🔧 Configuration Test")
    print("=" * 30)
    print(f"ChromaDB Path: {get_chroma_db_path()}")
    print(f"Default Collection: {get_best_collection_name()}")
    print()
    
    print("📚 Available Collections:")
    collections = list_available_collections()
    for name, count in collections:
        print(f"  - {name}: {count} documents")
    print()
    
    print("🛒 Available Products:")
    products = get_available_products()
    for display_name, info in products.items():
        print(f"  - {display_name}: {info['document_count']} docs ({info['collection']})")
    print()
    
    print("🔍 Product Metadata:")
    for product_key in PRODUCT_COLLECTIONS.keys():
        if product_key != 'mixed':
            metadata = get_product_collection_metadata(product_key)
            print(f"  - {product_key}: {metadata}") 