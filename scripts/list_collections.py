#!/usr/bin/env python3
"""
List all ChromaDB collections and their details
"""
import chromadb

def list_collections():
    client = chromadb.PersistentClient(path='./chroma_db')
    collections = client.list_collections()
    
    print("📚 Available ChromaDB Collections:")
    print("=" * 50)
    
    for collection_info in collections:
        collection = client.get_collection(collection_info.name)
        count = collection.count()
        
        # Get sample metadata to understand the product type
        if count > 0:
            sample = collection.peek(limit=1)
            metadata = sample['metadatas'][0] if sample['metadatas'] else {}
            product = metadata.get('product', 'unknown')
            category = metadata.get('category', 'unknown')
        else:
            product = 'unknown'
            category = 'unknown'
        
        print(f"🔍 Collection: {collection_info.name}")
        print(f"   Documents: {count}")
        print(f"   Product: {product}")
        print(f"   Category: {category}")
        print()

if __name__ == "__main__":
    list_collections() 