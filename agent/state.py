from typing import TypedDict, List, Dict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    requirements: Dict[str, List[str]]  # category -> list of requirements 
    pending: str | None  # NEW: holds a requirement waiting for confirmation

# Add this key to your initial state
state: AgentState = {
    "messages": [],
    "requirements": {
        "functional": [],
        "performance": [],
        "security": [],
        "integration": [],
        "budget": []
    },
    "pending": None  # NEW: holds a requirement waiting for confirmation
} 