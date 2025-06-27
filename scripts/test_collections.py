#!/usr/bin/env python3
"""
Test script to verify product-specific collection functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from requirements_analyzer import analyze_single_requirement_specialist
from config import get_available_products, get_collection_name

async def test_collection_switching():
    """Test that different collections return different evidence"""
    
    print("🔍 Testing Product-Specific Collection Functionality")
    print("=" * 60)
    
    # Get available products
    products = get_available_products()
    print(f"Available products: {list(products.keys())}")
    
    # Test requirement that should get different evidence from different collections
    test_requirement = {
        "text": "The system must support real-time data processing",
        "priority": "obligatory"
    }
    
    print(f"\nTest Requirement: {test_requirement['text']}")
    print("\n" + "="*60)
    
    # Test each available product collection
    for display_name, info in products.items():
        product_key = info['product_key']
        collection_name = info['collection']
        
        print(f"\n🔍 Testing with {display_name} ({collection_name})")
        print(f"   Product key: {product_key}")
        print(f"   Documents: {info['document_count']}")
        
        try:
            # Analyze the requirement using this specific collection
            result = await analyze_single_requirement_specialist(
                test_requirement, 
                "performance",  # Category
                collection_name  # Use specific collection
            )
            
            if result and not result.get('error'):
                print(f"   ✅ Analysis successful")
                print(f"   📊 Feasibility: {result.get('feasibility_score', 'N/A')}")
                print(f"   🎯 Support: {result.get('support_grade', 'N/A')}")
                
                # Show first source to verify it's from the right collection
                sources = result.get('sources', [])
                if sources:
                    print(f"   📄 First source: {sources[0]}")
                
                # Show evidence queries used
                queries = result.get('evidence_queries', [])
                if queries:
                    print(f"   🔎 Evidence queries: {', '.join(queries[:2])}")
                    
            else:
                print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("-" * 40)
    
    print("\n✅ Collection switching test complete!")

if __name__ == "__main__":
    asyncio.run(test_collection_switching()) 