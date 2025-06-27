"""
FastAPI server for requirements analysis.
Wraps the improved_analyzer_requirements_v3.py script as a web service.
"""

import asyncio
import os
import sys
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the analyzer functions
from requirements_analyzer_enhanced import (
    analyze_category_requirements_specialist,
    generate_final_specialist_report,
    test_model_connections,
    analyze_single_requirement_specialist
)

app = FastAPI(
    title="Requirements Analysis API",
    description="Evidence-based requirements analysis using specialist models",
    version="1.0.0"
)

# Add CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequirementsInput(BaseModel):
    """Input model for requirements analysis"""
    requirements: Dict[str, List[str]] = {
        "functional": [],
        "performance": [],
        "security": [],
        "integration": [],
        "budget": []
    }
    product: str = None  # Optional product key for collection selection

class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    report: str
    status: str
    requirements_analyzed: int
    product_used: str = None
    collection_used: str = None

def convert_requirements_format(requirements_dict: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
    """Convert simple string list to format expected by analyzer"""
    converted = {}
    for category, req_list in requirements_dict.items():
        converted[category] = []
        for req_text in req_list:
            # Default priority for all requirements
            priority = "obligatory" if any(word in req_text.lower() for word in ["must", "required", "shall"]) else "desired"
            converted[category].append({
                "text": req_text,
                "priority": priority
            })
    return converted

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Requirements Analysis API is running",
        "status": "healthy",
        "endpoints": ["/analyze", "/health"]
    }

@app.get("/health")
async def health_check():
    """Health check with model connectivity test"""
    try:
        models_ok = test_model_connections()
        return {
            "status": "healthy" if models_ok else "degraded",
            "models_connected": models_ok,
            "message": "All systems operational" if models_ok else "Model connection issues detected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "models_connected": False,
            "error": str(e)
        }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_requirements(input_data: RequirementsInput):
    """
    Analyze requirements using specialist models and evidence-based approach.
    
    This endpoint:
    1. Converts requirements to the format expected by the analyzer
    2. Selects the appropriate collection based on product
    3. Runs specialist-driven analysis for each category
    4. Generates a comprehensive final report
    5. Returns the markdown report
    """
    try:
        # Extract requirements and product from input
        requirements_dict = input_data.requirements
        selected_product = input_data.product
        
        # Determine which collection to use
        from config import get_collection_name, get_product_collection_metadata
        
        if selected_product:
            collection_name = get_collection_name(selected_product)
            collection_metadata = get_product_collection_metadata(selected_product)
            if not collection_metadata.get('available', False):
                from config import get_available_products
                raise HTTPException(
                    status_code=400,
                    detail=f"Product '{selected_product}' collection not available or empty. Available products: {list(get_available_products().keys())}"
                )
        else:
            collection_name = get_collection_name()
            selected_product = "auto-detected"
            collection_metadata = {"collection_name": collection_name}
        
        print(f"📊 Using collection: {collection_name} (product: {selected_product})")
        
        # Convert input to expected format
        requirements_formatted = convert_requirements_format(requirements_dict)
        
        # Count total requirements
        total_requirements = sum(len(reqs) for reqs in requirements_formatted.values())
        
        if total_requirements == 0:
            raise HTTPException(
                status_code=400, 
                detail="No requirements provided. Please add at least one requirement."
            )
        
        # Check minimum requirements threshold (recommended: at least 3)
        if total_requirements < 3:
            print(f"⚠️  Warning: Only {total_requirements} requirements provided. Analysis may be limited.")
        
        print(f"📊 Starting analysis of {total_requirements} requirements...")
        print(f"   Categories with requirements: {[cat for cat, reqs in requirements_formatted.items() if reqs]}")
        print(f"   Using collection: {collection_name}")
        
        # Analyze each category that has requirements
        category_tasks = []
        for category, reqs in requirements_formatted.items():
            if reqs:  # Only analyze categories with requirements
                print(f"   Scheduling {category}: {len(reqs)} requirements")
                # Pass collection_name to the analyzer function
                task = analyze_category_requirements_specialist(category, reqs, collection_name)
                category_tasks.append(task)
        
        if not category_tasks:
            raise HTTPException(
                status_code=400,
                detail="No valid requirements found to analyze."
            )
        
        print(f"🔄 Running {len(category_tasks)} category analyses in parallel...")
        
        # Run all category analyses in parallel
        category_results = await asyncio.gather(*category_tasks)
        
        # Filter out None results
        valid_results = [result for result in category_results if result is not None]
        
        if not valid_results:
            raise HTTPException(
                status_code=500,
                detail="All category analyses failed. Please check the server logs."
            )
        
        # Generate final report
        print("📝 Generating final comprehensive report...")
        final_report = await generate_final_specialist_report(valid_results)
        
        print(f"✅ Analysis complete! Generated {len(final_report)} character report")
        
        return AnalysisResponse(
            report=final_report,
            status="success",
            requirements_analyzed=total_requirements,
            product_used=selected_product,
            collection_used=collection_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.post("/analyze-single")
async def analyze_single_requirement(requirement: str, category: str = "functional"):
    """
    Analyze a single requirement (for testing purposes).
    """
    try:
        req_dict = {"text": requirement, "priority": "obligatory"}
        result = await analyze_single_requirement_specialist(req_dict, category)
        
        return {
            "requirement": requirement,
            "category": category,
            "analysis": result,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Single requirement analysis failed: {str(e)}"
        )

if __name__ == "__main__":
    print("Starting Requirements Analysis API Server...")
    print("Endpoints:")
    print("  - GET  /         : API info")
    print("  - GET  /health   : Health check")
    print("  - POST /analyze  : Full requirements analysis")
    print("  - POST /analyze-single : Single requirement analysis")
    print("\nAccess API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 