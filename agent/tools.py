from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Dict, List
from langchain_ibm import WatsonxLLM
from langchain.prompts import PromptTemplate
import os
from agent.llm import analysis_llm

# Save Requirement Tool (no longer used directly)
class SaveRequirementInput(BaseModel):
    requirement: str = Field(description="The requirement to save")
    category: str = Field(description="Category: functional, performance, security, integration, or budget")

def save_requirement_wrapper(requirement: str, category: str, **kwargs) -> dict:
    current_requirements = kwargs.get("requirements", {
        "functional": [],
        "performance": [],
        "security": [],
        "integration": [],
        "budget": []
    })
    if category not in current_requirements:
        return {"output": f"Invalid category: {category}", "requirements": current_requirements}
    current_requirements[category].append(requirement)
    msg = f"✓ Saved to {category}: '{requirement}'"
    return {"output": msg, "requirements": current_requirements}

save_requirement = StructuredTool(
    name="save_requirement",
    description="(Deprecated: use validate_requirement + confirm_save)",
    func=save_requirement_wrapper,
    args_schema=SaveRequirementInput
)

# Status Tool (formerly Check Completeness)
class EmptyInput(BaseModel):
    pass

def status_func(**kwargs) -> dict:
    current_requirements = kwargs.get("requirements", {
        "functional": [],
        "performance": [],
        "security": [],
        "integration": [],
        "budget": []
    })
    filled = {cat: len(reqs) for cat, reqs in current_requirements.items() if reqs}
    missing = [cat for cat, reqs in current_requirements.items() if not reqs]
    total = sum(len(reqs) for reqs in current_requirements.values())
    is_ready = total >= 3
    if is_ready and not missing:
        recommendation = (
            "All requirement categories have at least one entry. You can proceed to analysis, "
            "but consider adding more requirements for better coverage."
        )
    elif is_ready and missing:
        recommendation = (
            f"You have enough requirements to analyze, but you are missing: {', '.join(missing)}. "
            "Consider adding these for a more complete analysis."
        )
    elif not is_ready:
        recommendation = (
            f"You need at least 3 requirements to analyze. Currently missing: {', '.join(missing)}. "
            "Add more requirements to proceed."
        )
    else:
        recommendation = "Add more requirements to proceed."
    return {
        "output": recommendation,
        "missing_categories": missing,
        "filled_categories": filled,
        "is_ready_for_analysis": is_ready,
        "total_requirements": total,
        "requirements": current_requirements,
        "recommendation": recommendation
    }

status = StructuredTool(
    name="status",
    func=status_func,
    description="Use for ANY question about requirements status, completeness, missing categories, or readiness for analysis.",
    args_schema=EmptyInput
)

# Analyze Requirements Tool
def analyze_requirements_func(**kwargs) -> dict:
    current_requirements = kwargs.get("requirements", {
        "functional": [],
        "performance": [],
        "security": [],
        "integration": [],
        "budget": []
    })
    if not current_requirements or all(not reqs for reqs in current_requirements.values()):
        return {"output": "No requirements found.", "requirements": current_requirements}
    total = sum(len(reqs) for reqs in current_requirements.values())
    if total < 3:
        return {"output": "You need at least 3 requirements to analyze.", "requirements": current_requirements}
    return {"output": f"Analysis complete. Total requirements: {total} (stub).", "requirements": current_requirements}

analyze_requirements = StructuredTool(
    name="analyze_requirements",
    func=analyze_requirements_func,
    description="Use this tool to analyze the gathered requirements from different stakeholder perspectives.",
    args_schema=EmptyInput
)

# --- NEW: Validate Requirement Tool (no auto-save, asks for confirmation) ---
class ValidateRequirementInput(BaseModel):
    requirement: str = Field(description="The requirement to validate")
    category: str = Field(description="Category: functional, performance, security, integration, or budget")

def validate_requirement_func(requirement: str, category: str, **kwargs) -> dict:
    # Valid categories only
    VALID_CATEGORIES = ["functional", "performance", "security", "integration", "budget"]
    
    # Map invalid categories to valid ones
    category_mapping = {
        "metadata store": "integration",
        "database": "integration", 
        "data": "integration",
        "storage": "integration",
        "infrastructure": "integration",
        "technical": "integration",
        "business": "functional",
        "user": "functional",
        "ui": "functional",
        "ux": "functional",
        "cost": "budget",
        "financial": "budget",
        "money": "budget",
        "auth": "security",
        "authentication": "security",
        "authorization": "security",
        "privacy": "security"
    }
    
    # Normalize and map category
    category_lower = category.lower().strip()
    if category_lower not in VALID_CATEGORIES:
        mapped_category = category_mapping.get(category_lower, "functional")  # Default to functional
        print(f"Mapped invalid category '{category}' to '{mapped_category}'")
        category = mapped_category
    
    # Send only the requirement text to minimize tokens
    prompt = f"Is this requirement clear and specific? '{requirement}'\nAnswer only: clear OR vague"
    try:
        feedback = analysis_llm.invoke(prompt).strip()
        
        # More precise detection logic
        feedback_lower = feedback.lower()
        
        # Look for clear indicators at the start of the response
        if feedback_lower.startswith('clear') or feedback_lower.startswith('.clear') or 'answer: clear' in feedback_lower:
            result = "This requirement is clear and specific."
        elif feedback_lower.startswith('vague') or 'answer: vague' in feedback_lower:
            result = "This requirement is vague. You might get better results if you make it more precise. For example: 'The system must use PostgreSQL or DB2 as the primary database for metadata storage.'"
        else:
            # Fallback: check for keywords in first 50 characters 
            first_part = feedback_lower[:50]
            if 'clear' in first_part and 'vague' not in first_part:
                result = "This requirement is clear and specific."
            elif 'vague' in first_part:
                result = "This requirement is vague. You might get better results if you make it more precise. For example: 'The system must use PostgreSQL or DB2 as the primary database for metadata storage.'"
            else:
                # Default to clear if unclear response
                result = "This requirement is clear and specific."
                
        print(f"DEBUG - Raw feedback: '{feedback[:100]}...'" if len(feedback) > 100 else f"DEBUG - Raw feedback: '{feedback}'")
        print(f"DEBUG - Classified as: {'CLEAR' if 'clear and specific' in result else 'VAGUE'}")
        
    except Exception as e:
        return {
            "output": f"Validation failed: {e}",
            "pending": None
        }
    return {
        "output": result,
        "pending": {"text": requirement, "category": category}
    }

validate_requirement = StructuredTool(
    name="validate_requirement",
    description="Check if a requirement is clear and specific, and ask the user if it should be saved.",
    func=validate_requirement_func,
    args_schema=ValidateRequirementInput
)

# --- NEW: Confirm Save Tool ---
class ConfirmSaveInput(BaseModel):
    decision: str = Field(description="'yes' to store the requirement in memory, anything else = discard")

def confirm_save_func(decision: str, **kwargs):
    # Always allow confirm_save when called with a decision
    # The __triggered_by__ check was too restrictive
    pending = kwargs.get("pending")
    state   = kwargs.get("requirements", {})
    
    if not pending:
        return {"output": "There is nothing waiting to be saved.", "pending": None, "requirements": state}
    
    if decision.lower().startswith("y"):
        state[pending["category"]].append(pending["text"])
        msg = f"✓ Saved to {pending['category']}: '{pending['text']}'"
    else:
        msg = "Okay, not saved."
    
    return {"output": msg, "pending": None, "requirements": state}

confirm_save = StructuredTool(
    name="confirm_save",
    func=confirm_save_func,
    description=(
        "Call this **only** immediately after the *user* has replied "
        "'yes' or 'no' to the question 'Save this requirement?'.\n"
        "Never guess the user's decision."
    ),
    args_schema=ConfirmSaveInput,
) 