import os
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai import APIClient
from agent.tools import save_requirement, status, analyze_requirements, validate_requirement, confirm_save
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, SystemMessage, AIMessage
from agent.state import AgentState
from langgraph.graph import StateGraph, END
from typing import Dict, List, Any
from langchain.globals import set_debug

# Load environment variables from .env file
load_dotenv()

# Load credentials from environment variables
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_ENDPOINT = os.getenv("WATSONX_ENDPOINT", "https://us-south.ml.cloud.ibm.com")

print(f"API Key loaded: {'Yes' if os.getenv('WATSONX_API_KEY') else 'No'}")

credentials = {
    "url": WATSONX_ENDPOINT,
    "api_key": WATSONX_API_KEY
}

client = APIClient(
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID
)

set_debug(True)

def is_strictly_clear_and_specific(feedback: str) -> bool:
    """Return True if feedback is exactly 'this requirement is clear and specific.' (case-insensitive, stripped)."""
    return feedback.strip().lower() == "this requirement is clear and specific."

system_prompt = (
    "You are a requirements gathering and analysis assistant.\n"
    "\n"
    "CRITICAL: You MUST use tools to answer questions. Never answer without calling the appropriate tool first.\n"
    "\n"
    "TOOL USAGE RULES:\n"
    "- For ANY question about status, completeness, missing items, or what's needed: ALWAYS call the 'status' tool. This includes questions like 'Which categories are missing?', 'Are there any empty categories?', 'What do I need to add?', 'Can I analyze now?', and 'Do I have enough requirements to proceed?'.\n"
    "- For saving requirements: ALWAYS call 'validate_requirement' FIRST for any new requirement. Only call 'save_requirement' if the requirement is clear and specific.\n"
    "- For analysis requests: ALWAYS call 'analyze_requirements'\n"
    "- NEVER provide status information or summaries without first calling the 'status' tool and using its output.\n"
    "- NEVER try to infer or state the status, completeness, or missing categories yourself. ALWAYS rely on the 'status' tool for this information.\n"
    "\n"
    "# FEW-SHOT TOOL CALL EXAMPLES\n"
    "<assistant>\n{\"tool\":\"status\",\"args\":{}}\n<tool>\nYou need at least 3 requirements to analyze.\n<user>\nCan I analyze my requirements now?\n<assistant>\n{\"tool\":\"status\",\"args\":{}}\n<tool>\nYou have enough requirements to analyze, but you are missing: integration, budget.\n<user>\nAdd a functional requirement: The system must allow user registration.\n<assistant>\n{\"tool\":\"validate_requirement\",\"args\":{\"requirement\":\"The system must allow user registration.\"}}\n<tool>\nThis requirement is clear and specific.\n<assistant>\n{\"tool\":\"save_requirement\",\"args\":{\"requirement\":\"The system must allow user registration.\",\"category\":\"functional\"}}\n<tool>\n✓ Saved to functional: 'The system must allow user registration.'\n"
    "<user>\nAdd a functional requirement: The system should be fast.\n<assistant>\n{\"tool\":\"validate_requirement\",\"args\":{\"requirement\":\"The system should be fast.\"}}\n<tool>\nThis requirement is vague. You might get better results if you make it more precise. For example: 'The system must respond to user actions within 2 seconds under normal load.'\n"
    "\n"
    "Examples:\n"
    "- User: 'Which categories are missing?' → You MUST call status\n"
    "- User: 'What do I need to add?' → You MUST call status\n"
    "- User: 'Can I analyze now?' → You MUST call status\n"
    "- User: 'Are there any empty categories?' → You MUST call status\n"
    "- User: 'Do I have enough requirements to proceed?' → You MUST call status\n"
    "\n"
    "MODES:\n"
    "1. GATHERING MODE: Help users formulate and save requirements\n"
    "2. ANALYSIS MODE: Analyze requirements from stakeholder perspectives\n"
    "RULES:\n"
    "- Use validate_requirement tool for any new requirement, then save_requirement only if clear.\n"
    "- Ask clarifying questions for vague inputs\n"
    "- Always end with 'What other requirements should we consider?'\n"
    "- Need minimum 3 requirements before analysis\n"
    "- Use analyze_requirements tool only when explicitly asked\n"
    "Current mode: GATHERING MODE."
    "\n"
    "# EXAMPLES OF CLEAR VS. VAGUE REQUIREMENTS\n"
    "Clear: 'The system must respond to user actions within 2 seconds under normal load.'\n"
    "Clear: 'All data must be encrypted in transit and at rest.'\n"
    "Vague: 'The system should be fast.'\n"
    "Vague: 'The app should be user-friendly.'\n"
    "Vague: 'The system should be secure.'\n"
    "For vague requirements, suggest a more precise version and do NOT call save_requirement.\n"
    "- Call validate_requirement when the user states a possible requirement.\n"
    "- After that, wait:\n"
    "  • \"yes\" → confirm_save(decision=\"yes\")\n"
    "  • \"no\"  → confirm_save(decision=\"no\")\n"
    "  • another requirement-like sentence → validate_requirement again\n"
    "- Never call save_requirement directly unless the user explicitly asks.\n"
    "- Only the **user** decides whether to save; never assume their answer.\n"
    "\n"
    "IMPORTANT: After confirm_save is called, do NOT call any more tools. Simply acknowledge the action taken."
)

chat = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=credentials["url"],
    project_id=WATSONX_PROJECT_ID,
    params={"temperature": 0.3, "max_tokens": 500},
    watsonx_client=client
)

# confirm_save must be last in the list to reduce model eagerness to call it
all_tools = [save_requirement, status, analyze_requirements, validate_requirement, confirm_save]
chat_with_tools = chat.bind_tools(all_tools)

def compact_history(msgs, window=2, include_system=None):
    """Only keep HumanMessage and AIMessage for LLM context, not ToolMessage"""
    relevant_msgs = [m for m in msgs if isinstance(m, (HumanMessage, AIMessage))]
    
    # Auto-detect if we need system prompt (first interaction)
    if include_system is None:
        include_system = len(relevant_msgs) <= 1
    
    # Take the last 'window' messages
    windowed_msgs = relevant_msgs[-window:] if len(relevant_msgs) > window else relevant_msgs
    
    # Return system prompt + windowed messages only if needed
    if include_system:
        return [SystemMessage(content=system_prompt)] + windowed_msgs
    else:
        return windowed_msgs

def llm_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    requirements = state.get("requirements", {})
    
    if state.get("__end__"):
        return state
    
    # If last message is confirm_save or save_requirement, only allow a natural reply, then halt
    if (len(messages) >= 2 and isinstance(messages[-1], ToolMessage) and hasattr(messages[-1], "tool_name") and messages[-1].tool_name in ("confirm_save", "save_requirement")):
        truncated_messages = compact_history(messages)
        response = chat.invoke(truncated_messages)
        return {
            "messages": state["messages"] + [response],
            "requirements": state.get("requirements", {}),
            "__end__": True
        }
    # Break the status loop - after status call, reply normally without tools and halt
    if (len(messages) >= 2 and isinstance(messages[-1], ToolMessage) and getattr(messages[-1], "tool_name", "") == "status"):
        truncated = compact_history(messages)
        response = chat.with_config({"allowed_tool_names": []}).invoke(truncated)
        return {
            "messages": messages + [response],
            "requirements": requirements,
            "__end__": True
        }
    
    # Special handling for compound requirements - split them and process first one
    if messages and isinstance(messages[-1], HumanMessage):
        user_content = messages[-1].content.strip()
        
        # Check if this looks like compound requirements (contains periods and requirement words)
        requirement_indicators = ["must", "should", "shall", "needs to", "support", "allow", "available", "export"]
        if (any(indicator in user_content.lower() for indicator in requirement_indicators) and 
            len([s for s in user_content.split('.') if s.strip()]) > 1):
            
            # Split into sentences and process the first one that looks like a requirement
            sentences = [s.strip() for s in user_content.split('.') if s.strip()]
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in requirement_indicators):
                    # Process this sentence as a requirement
                    import re
                    cat_match = re.search(r"(functional|performance|security|integration|budget)", sentence, re.IGNORECASE)
                    category = cat_match.group(1).lower() if cat_match else "functional"
                    
                    # Auto-detect category based on content
                    if any(word in sentence.lower() for word in ["database", "db2", "postgres", "sql", "integration", "connect"]):
                        category = "integration"
                    elif any(word in sentence.lower() for word in ["latency", "response", "performance", "speed", "time"]):
                        category = "performance"
                    elif any(word in sentence.lower() for word in ["security", "encrypt", "auth", "permission"]):
                        category = "security"
                    elif any(word in sentence.lower() for word in ["export", "report", "download", "format", "json", "csv"]):
                        category = "functional"
                    
                    validation = validate_requirement.func(requirement=sentence, category=category)
                    
                    if "clear and specific" in validation["output"].lower():
                        prompt = "Save this requirement? (yes / no)"
                        ai_msg = AIMessage(content=prompt)
                        return {
                            "messages": state["messages"] + [ai_msg],
                            "requirements": requirements,
                            "pending": {"text": sentence, "category": category},
                            "__end__": True
                        }
                    elif "vague" in validation["output"].lower():
                        coaching = "This requirement is too vague. Please make it more specific. For example: 'The system must respond to user actions within 2 seconds under normal load.'\n\nSave this vague requirement anyway? (yes / no)"
                        ai_msg = AIMessage(content=coaching)
                        return {
                            "messages": state["messages"] + [ai_msg], 
                            "requirements": requirements,
                            "pending": {"text": sentence, "category": category},
                            "__end__": True
                        }
                    break
    
    truncated_messages = compact_history(messages)
    response = chat_with_tools.invoke(truncated_messages)
    
    if getattr(response, "tool_calls", []):
        return {
            "messages": state["messages"] + [response],
            "requirements": state.get("requirements", {})
        }
    
    # Check if user is responding to a save confirmation question
    if (len(messages) >= 2 and 
        isinstance(messages[-2], ToolMessage) and 
        getattr(messages[-2], "tool_name", "") == "validate_requirement" and
        "clear and specific" in messages[-2].content.lower() and
        isinstance(messages[-1], HumanMessage)):
        
        user_response = messages[-1].content.strip().lower()
        if user_response in ["yes", "y", "save", "ok"]:
            decision = "yes"
        elif user_response in ["no", "n", "don't save", "skip"]:
            decision = "no"
        else:
            decision = None
            
        if decision:
            # Call confirm_save with the user's decision
            output = confirm_save.func(decision=decision, requirements=requirements, pending=state.get("pending"), __triggered_by__="HumanMessage")
            tool_msg = ToolMessage(
                tool_name="confirm_save",
                tool_call_id="auto-confirm",
                content=output["output"]
            )
            return {
                "messages": messages + [response, tool_msg],
                "requirements": output.get("requirements", requirements),
                "pending": output.get("pending"),
                "__end__": True
            }
    
    # Fallback: keyword heuristic (for cases where LLM doesn't call tools)
    if messages and isinstance(messages[-1], HumanMessage):
        user_content = messages[-1].content.strip().lower()
        requirement_keywords = [
            "requirement", "must", "should", "needs to", "shall", "allow", "support"
        ]
        force_status_keywords = [
            "missing", "empty", "analyze", "analyse", "enough", "categories", "what do i need", "how many"
        ]
        
        # Only trigger requirement processing for single, clear requirement statements
        if (any(req_kw in user_content for req_kw in requirement_keywords) and 
            not any(status_kw in user_content for status_kw in force_status_keywords) and
            len([s for s in user_content.split('.') if s.strip()]) == 1):  # Single sentence only
            
            # Extract requirement and category if possible
            import re
            req_match = re.search(r"requirement:?(.*)", messages[-1].content, re.IGNORECASE)
            requirement = req_match.group(1).strip() if req_match else messages[-1].content.strip()
            
            # Try to extract category
            cat_match = re.search(r"(functional|performance|security|integration|budget)", messages[-1].content, re.IGNORECASE)
            category = cat_match.group(1).lower() if cat_match else "functional"
            
            # Validate requirement
            validation = validate_requirement.func(requirement=requirement, category=category)
            feedback = validation["output"].strip().lower()
            tool_msg = ToolMessage(
                tool_name="validate_requirement",
                tool_call_id="fallback-validate",
                content=validation["output"]
            )
            new_messages = state["messages"] + [tool_msg]
            return {"messages": new_messages, "requirements": requirements, "pending": validation.get("pending")}
        elif any(status_kw in user_content for status_kw in force_status_keywords):
            # Directly call status tool
            status_output = status.func(requirements=requirements)
            tool_msg = ToolMessage(
                tool_name="status",
                tool_call_id="forced-by-keyword",
                content=status_output["output"]
            )
            new_messages = state["messages"] + [tool_msg]
            return {"messages": new_messages, "requirements": status_output["requirements"]}
    
    # Otherwise, use LLM response as-is
    new_messages = state["messages"] + [response]
    return {"messages": new_messages, "requirements": state.get("requirements", {})}

def tool_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return state
    tool_outputs = []
    requirements = state.get("requirements", {
        "functional": [],
        "performance": [],
        "security": [],
        "integration": [],
        "budget": []
    })
    pending = state.get("pending", None)
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_kwargs = {
            "requirements": requirements,
            "pending": pending
        }
        if tool_name == "validate_requirement":
            requirement = tool_args.get("requirement")
            category = tool_args.get("category", "functional")
            output = validate_requirement.func(requirement=requirement, category=category, **tool_kwargs)
            
            # Don't add validation message to chat - just use it for logic
            # tool_msg = ToolMessage(
            #     tool_name="validate_requirement",
            #     tool_call_id=tool_call["id"], 
            #     content=output["output"]
            # )
            # tool_outputs.append(tool_msg)
            
            if "clear and specific" in output["output"].lower():
                # ✔ requirement is clear → ask user to save (no duplicate message)
                prompt = "Save this requirement? (yes / no)"
                tool_outputs.append(AIMessage(content=prompt))
                pending = {"text": requirement, "category": category}
                return {
                    "messages": messages + tool_outputs,
                    "requirements": requirements,
                    "pending": pending,
                    "__end__": True      # stop – wait for user's yes/no
                }
            elif "vague" in output["output"].lower():
                # ✗ requirement is vague → give coaching with save option
                coaching = "This requirement is too vague. Please make it more specific. For example: 'The system must respond to user actions within 2 seconds under normal load.'\n\nSave this vague requirement anyway? (yes / no)"
                tool_outputs.append(AIMessage(content=coaching))
                pending = {"text": requirement, "category": category}
                return {
                    "messages": messages + tool_outputs,
                    "requirements": requirements,
                    "pending": pending,
                    "__end__": True      # stop – wait for user to confirm or reject
                }
            
            pending = output.get("pending", pending)
        elif tool_name in ("confirm_save", "save_requirement"):
            triggered_by = type(messages[-2]).__name__ if len(messages) > 1 else ""
            if tool_name == "confirm_save":
                output = confirm_save.func(**tool_args, **tool_kwargs, __triggered_by__=triggered_by)
            else:
                requirement = tool_args.get("requirement")
                category = tool_args.get("category", "functional")
                validation = validate_requirement.func(requirement=requirement, category=category)
                feedback = validation["output"].strip()
                if is_strictly_clear_and_specific(feedback):
                    output = save_requirement.func(
                        requirement=requirement,
                        category=category,
                        requirements=requirements
                    )
                else:
                    output = {"output": feedback, "requirements": requirements, "pending": pending}
            if "ignored" not in output["output"].lower():
                tool_msg = ToolMessage(
                    tool_name=tool_name,
                    tool_call_id=tool_call["id"], 
                    content=output["output"]
                )
                tool_outputs.append(tool_msg)
            requirements = output.get("requirements", requirements)
            pending = output.get("pending", pending)
            return {
                "messages": messages + tool_outputs, 
                "requirements": requirements, 
                "pending": pending, 
                "__end__": True
            }
        elif tool_name == "status":
            output = status.func(requirements=requirements)
            requirements = output.get("requirements", requirements)
            tool_msg = ToolMessage(
                tool_name="status",
                tool_call_id=tool_call["id"], 
                content=str(output["output"])
            )
            tool_outputs.append(tool_msg)
        elif tool_name == "analyze_requirements":
            output = analyze_requirements.func(requirements=requirements)
            requirements = output.get("requirements", requirements)
            tool_msg = ToolMessage(
                tool_name="analyze_requirements",
                tool_call_id=tool_call["id"], 
                content=str(output["output"])
            )
            tool_outputs.append(tool_msg)
        else:
            tool_msg = ToolMessage(
                tool_name=tool_name,
                tool_call_id=tool_call["id"], 
                content=f"Unknown tool: {tool_name}"
            )
            tool_outputs.append(tool_msg)
    new_messages = messages + tool_outputs
    return {"messages": new_messages, "requirements": requirements, "pending": pending}

# Build the StateGraph
workflow = StateGraph(AgentState)
workflow.add_node("llm", llm_node)
workflow.add_node("tools", tool_node)

# Edges
workflow.set_entry_point("llm")

# Replace the unconditional edge with a conditional one that checks __end__
workflow.add_conditional_edges(
    "tools",
    lambda s: END if s.get("__end__") else "llm",
    {"llm": "llm", END: END}
)

# Keep the existing conditional on "llm"
workflow.add_conditional_edges(
    "llm",
    lambda s: END if s.get("__end__") else ("tools" if s.get("messages") and hasattr(s["messages"][-1], "tool_calls") and s["messages"][-1].tool_calls else END),
    {"tools": "tools", END: END}
)

app = workflow.compile()

if __name__ == "__main__":
    # Initialize state with empty requirements
    state: AgentState = {
        "messages": [],
        "requirements": {
            "functional": [],
            "performance": [],
            "security": [],
            "integration": [],
            "budget": []
        },
        "pending": None
    }
    
    test_prompts = [
        "The system needs user authentication",
        "Which requirement categories are missing?",
        "Analyze my requirements",
        "Echo the following: hello world"
    ]
    
    for prompt in test_prompts:
        print(f"\n=== Testing prompt: {prompt} ===")
        # Start a new turn with the user's message
        state["messages"] = [HumanMessage(content=prompt)]
        result = app.invoke(state)
        print("Final agent messages:")
        for msg in result["messages"]:
            print(f"- {getattr(msg, 'content', repr(msg))}")
        print("Current requirements state:")
        print(result.get("requirements", {}))
        state = result  # Carry forward full state