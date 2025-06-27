#!/usr/bin/env python3
"""
Check what metadata ChromaDB currently has
"""
import chromadb
import json

def check_metadata():
    client = chromadb.PersistentClient(path='./chroma_db')
    collection = client.get_collection('documents')
    
    print(f"Collection: {collection.name}")
    print(f"Count: {collection.count()}")
    
    # Get a few samples to check metadata
    results = collection.peek(limit=5)
    
    if results['metadatas'] and results['metadatas'][0]:
        print(f"Metadata keys: {list(results['metadatas'][0].keys())}")
        
        print("\nAll metadata samples:")
        for i, metadata in enumerate(results['metadatas'][:5]):
            print(f"{i+1}. {json.dumps(metadata, indent=2)}")
    else:
        print("No metadata found")
    
    # Show sample documents
    print("\nSample documents with metadata:")
    for i, (doc, metadata) in enumerate(zip(results['documents'][:3], results['metadatas'][:3])):
        print(f"{i+1}. Source: {metadata.get('source', 'N/A')}, Category: {metadata.get('category', 'N/A')}")
        print(f"   Text: {doc[:100]}...")
        print()

if __name__ == "__main__":
    check_metadata() 