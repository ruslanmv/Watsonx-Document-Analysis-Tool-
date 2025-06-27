#!/usr/bin/env python3
"""
Utility script to check ChromaDB collections and debug database issues.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import list_available_collections, get_best_collection_name, get_chroma_db_path
from agent.embeddings import get_embeddings

def main():
    print("🔍 ChromaDB Collection Inspector")
    print("=" * 50)
    
    db_path = Path(get_chroma_db_path())
    print(f"📂 Database path: {db_path.absolute()}")
    print(f"📁 Database exists: {db_path.exists()}")
    
    if not db_path.exists():
        print("\n❌ No ChromaDB found!")
        print("   Run: python scripts/ingest_docs.py --reset")
        return
    
    print(f"\n📊 Checking for collections...")
    collections = list_available_collections()
    
    if not collections:
        print("❌ No collections with data found!")
        print("   Available options:")
        print("   1. Run: python scripts/ingest_docs.py --reset")
        print("   2. Check if documents exist in ./documents folder")
        return
    
    print(f"✅ Found {len(collections)} collections with data:")
    for collection_name, doc_count in collections:
        print(f"   📚 '{collection_name}': {doc_count} documents")
    
    best_collection = get_best_collection_name()
    print(f"\n🎯 Auto-selected collection: '{best_collection}'")
    
    # Test connection to best collection
    try:
        from langchain_chroma import Chroma
        embeddings = get_embeddings()
        db = Chroma(
            collection_name=best_collection,
            embedding_function=embeddings,
            persist_directory=str(db_path)
        )
        
        # Test a simple query
        test_docs = db.similarity_search("system requirements", k=2)
        print(f"✅ Connection test successful: found {len(test_docs)} relevant documents")
        
        if test_docs:
            print(f"📖 Sample document preview:")
            sample_doc = test_docs[0]
            content_preview = sample_doc.page_content[:200] + "..." if len(sample_doc.page_content) > 200 else sample_doc.page_content
            print(f"   {content_preview}")
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        
    print(f"\n💡 To fix collection issues:")
    print(f"   • Ensure documents are in ./documents folder")
    print(f"   • Run: python scripts/ingest_docs.py --reset")
    print(f"   • Set COLLECTION_NAME environment variable if needed")

if __name__ == "__main__":
    main() 