import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress torch/streamlit compatibility warnings
import warnings
warnings.filterwarnings("ignore", message=".*torch.*")
warnings.filterwarnings("ignore", message=".*RuntimeError.*event loop.*")

import time
import requests
import json
import streamlit as st
import streamlit.components.v1
from agent.agent import app, AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent.tools import status
import datetime

st.set_page_config(page_title="Requirement Assistant", layout="wide")

# Session state initialization
if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "requirements": {
            "functional": [],
            "performance": [],
            "security": [],
            "integration": [],
            "budget": []
        }
    }
if "analysis_report" not in st.session_state:
    st.session_state.analysis_report = None
if "products" not in st.session_state:
    # Dynamically load available products from ChromaDB
    try:
        from config import get_available_products
        available_products = get_available_products()
        if available_products:
            product_options = []
            for display_name, info in available_products.items():
                product_options.append(f"📊 {display_name} ({info['document_count']} docs)")
            st.session_state.products = product_options
            st.session_state.product_mapping = available_products
        else:
            # Fallback to default if no products found
            st.session_state.products = ["📚 Demo: No product-specific collections found"]
            st.session_state.product_mapping = {}
    except Exception as e:
        # Fallback for any errors
        st.session_state.products = [
            "📊 DataFlow Nexus - Data Orchestration Platform", 
            "📈 IBM Cognos - Business Intelligence & Reporting",
            "🔍 Demo: Mixed Document Analysis"
        ]
        st.session_state.product_mapping = {}
if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = None
if "feedback_time" not in st.session_state:
    st.session_state.feedback_time = 0
if "pending_save" not in st.session_state:
    st.session_state.pending_save = False
if "selected_product_key" not in st.session_state:
    st.session_state.selected_product_key = None

def show_feedback(msg):
    """Show temporary feedback message"""
    st.session_state.feedback_message = msg
    st.session_state.feedback_time = time.time()
    st.session_state.pending_save = "vague" in msg.lower()
    st.rerun()

def process_user_input():
    """Process user input and invoke the agent"""
    user_input = st.session_state.pop("user_input_box", "")
    if not user_input.strip():
        return
    
    # Add user message to chat
    st.session_state.state["messages"].append(HumanMessage(content=user_input))
    
    # Process with agent
    result = app.invoke(st.session_state.state)
    st.session_state.state = result
    
    # Check for validation feedback
    last_tool_msg = None
    for msg in reversed(result["messages"]):
        if isinstance(msg, ToolMessage):
            if "vague" in getattr(msg, "content", "").lower() or "improve" in getattr(msg, "content", "").lower():
                last_tool_msg = msg.content
                break
    
    if last_tool_msg:
        show_feedback(last_tool_msg)
    
    st.rerun()

def remove_requirement(cat, idx):
    """Remove a requirement from the specified category"""
    if 0 <= idx < len(st.session_state.state["requirements"][cat]):
        removed_req = st.session_state.state["requirements"][cat].pop(idx)
        st.toast(f"✓ Removed requirement: {removed_req[:50]}...")
    st.rerun()

def analyze_requirements():
    """Call the FastAPI analyzer endpoint with product-specific collection"""
    try:
        with st.spinner("🔍 Running evidence-based analysis... This may take 30-60 seconds"):
            # Prepare the request payload
            payload = {
                "requirements": st.session_state.state["requirements"]
            }
            
            # Add product selection if available
            if st.session_state.selected_product_key:
                payload["product"] = st.session_state.selected_product_key
            
            response = requests.post(
                "http://localhost:8000/analyze",
                json=payload,
                timeout=120
            )
            if response.status_code == 200:
                # Parse JSON response
                result = response.json()
                st.session_state.analysis_report = result.get("report", "")
                requirements_count = result.get("requirements_analyzed", 0)
                product_used = result.get("product_used", "unknown")
                collection_used = result.get("collection_used", "unknown")
                
                st.toast(f"✅ Analysis complete! Analyzed {requirements_count} requirements using {product_used} ({collection_used}).")
            else:
                try:
                    error_detail = response.json().get("detail", response.text)
                except:
                    error_detail = response.text
                st.error(f"❌ Analysis failed: {response.status_code} - {error_detail}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Analysis server not running. Please start the FastAPI server first.")
        st.info("💡 Run: `poetry run python run_app.py --mode api` to start the server")
    except requests.exceptions.Timeout:
        st.error("❌ Analysis timed out. The server may be overloaded.")
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
    st.rerun()

# === MAIN UI LAYOUT ===

st.title("🤖 Requirement Assistant")

# Sidebar with intro and product selector
with st.sidebar:
    with st.expander("About this demo", expanded=True):
        st.markdown("""
        **Requirement Assistant** helps you:
        1. Enter requirements in natural language  
        2. Validate and categorize them
        3. Run evidence-based analysis against business documents
        4. Generate comprehensive requirement reports
        
        **Setup**: See README.md for document ingestion and configuration instructions.
        """)
    
    # Product selector
    selected_product = st.selectbox(
        "🛒 Select knowledge base",
        options=st.session_state.products,
        key="selected_product",
        help="Choose which product documentation to analyze against"
    )
    
    # Extract product key from selection and update state
    if st.session_state.product_mapping:
        # Find the matching product key
        selected_display_name = selected_product.split(" (")[0].replace("📊 ", "")
        for display_name, info in st.session_state.product_mapping.items():
            if display_name == selected_display_name:
                if st.session_state.selected_product_key != info['product_key']:
                    st.session_state.selected_product_key = info['product_key']
                    # Show info about the switch
                    st.info(f"📊 **Switched to {display_name}**: {info['document_count']} documents in collection '{info['collection']}'")
                break
    
    # Demo explanation based on actual data
    if st.session_state.product_mapping:
        total_products = len(st.session_state.product_mapping)
        total_docs = sum(info['document_count'] for info in st.session_state.product_mapping.values())
        st.info(f"💡 **Knowledge Base**: {total_products} product collections available with {total_docs} total documents. Requirements will be analyzed against the selected product's documentation.")
    else:
        st.info("💡 **Demo Mode**: Using fallback product list. Set up product-specific collections for full functionality.")
    
    # Add new product option (disabled in product-specific mode)
    if not st.session_state.product_mapping:
        if st.button("➕ Add Custom Scenario"):
            new_product = st.text_input("Scenario name:", key="new_product_input", placeholder="e.g., 'My Analytics Platform'")
            if new_product and new_product not in st.session_state.products:
                st.session_state.products.append(f"🎯 Custom: {new_product}")
                st.rerun()
    else:
        st.caption("💡 Use `scripts/ingest_docs.py --product <name>` to add new product collections")
    
    st.divider()
    
    # Requirements dashboard
    st.subheader("📋 Current Requirements")
    category_icons = {
        "functional": "🛠️",
        "performance": "⚡",
        "security": "🔒",
        "integration": "🔗", 
        "budget": "💰"
    }
    
    total_reqs = sum(len(reqs) for reqs in st.session_state.state["requirements"].values())
    st.metric("Total Requirements", total_reqs)
    
    for cat, reqs in st.session_state.state["requirements"].items():
        icon = category_icons.get(cat, "📝")
        with st.expander(f"{icon} {cat.capitalize()} ({len(reqs)})", expanded=len(reqs) > 0):
            if reqs:
                for idx, r in enumerate(reqs):
                    col1, col2 = st.columns([0.85, 0.15])
                    with col1:
                        st.write(f"• {r}")
                    with col2:
                        if st.button("🗑️", key=f"remove_{cat}_{idx}", help="Delete requirement"):
                            remove_requirement(cat, idx)
            else:
                st.write("*No requirements yet*")

# Main chat area
st.subheader("💬 Chat with the Agent")

# Action buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📊 Analyze Requirements", help="Run evidence-based analysis", type="primary"):
        analyze_requirements()
with col2:
    if st.button("📋 Show Status", help="Check requirement completeness"):
        requirements = st.session_state.state["requirements"]
        status_result = status.func(requirements=requirements)
        st.session_state.state["messages"].append(
            AIMessage(content=status_result["output"])
        )
        st.rerun()
with col3:
    if st.button("🗑️ Clear Chat", help="Clear chat history"):
        st.session_state.state["messages"] = []
        st.rerun()

# Chat history with modern chat UI
for msg in st.session_state.state["messages"]:
    if hasattr(msg, "content") and msg.content.strip():
        role = "user" if getattr(msg, "type", "") == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

# Check if we need to show Yes/No buttons for save confirmation
show_save_buttons = False
last_ai_message = None

# Only show buttons if the LAST message is asking for confirmation and there's no response yet
if len(st.session_state.state["messages"]) >= 1:
    last_msg = st.session_state.state["messages"][-1]
    
    # Debug information
    if hasattr(last_msg, "content") and hasattr(last_msg, "type"):
        msg_content = last_msg.content.lower()
        msg_type = last_msg.type
        
        # Debug display (remove this later)
        # st.write(f"Debug - Last message type: {msg_type}")
        # st.write(f"Debug - Last message content: {last_msg.content[:100]}...")
        # st.write(f"Debug - Has 'save this': {'save this' in msg_content}")
        # st.write(f"Debug - Has 'requirement': {'requirement' in msg_content}")
        # st.write(f"Debug - Has '(yes / no)': {'(yes / no)' in msg_content}")
    
    # Show buttons only if:
    # 1. Last message is AI asking for save confirmation
    # 2. There's no human response after it yet
    # Updated logic to catch both "Save this requirement?" and "Save this vague requirement anyway?"
    if (hasattr(last_msg, "content") and hasattr(last_msg, "type") and 
        last_msg.type == "ai" and "save this" in last_msg.content.lower() and 
        "requirement" in last_msg.content.lower() and 
        "(yes / no)" in last_msg.content.lower()):
        show_save_buttons = True
        last_ai_message = last_msg.content

# Show Yes/No buttons when save confirmation is needed
if show_save_buttons:
    st.markdown("---")
    st.write("**Please choose:**")
    
    col_yes, col_no, col_edit = st.columns(3)
    
    with col_yes:
        if st.button("✅ Yes, Save", key="btn_save_req", type="primary"):
            st.session_state.state["messages"].append(HumanMessage(content="yes"))
            result = app.invoke(st.session_state.state)
            st.session_state.state = result
            st.rerun()
    
    with col_no:
        if st.button("❌ No, Skip", key="btn_skip_req"):
            st.session_state.state["messages"].append(HumanMessage(content="no"))
            result = app.invoke(st.session_state.state)
            st.session_state.state = result
            st.rerun()
    
    with col_edit:
        if st.button("✏️ Let me rephrase", key="btn_edit_req"):
            # Remove the save confirmation AND the original user requirement to allow rephrasing
            messages = st.session_state.state["messages"]
            
            # Find and remove the save confirmation message (last AI message)
            if messages and hasattr(messages[-1], "type") and messages[-1].type == "ai":
                messages.pop()
            
            # Also remove the original user requirement message (should be a human message)
            # Go backwards to find the last human message
            for i in range(len(messages) - 1, -1, -1):
                if hasattr(messages[i], "type") and messages[i].type == "human":
                    messages.pop(i)
                    break
            
            st.toast("💬 Please enter your requirement again (you can rephrase it now)")
            st.rerun()

# Feedback bar for vague requirements (legacy - kept for backward compatibility)
feedback_displayed = False
if st.session_state.feedback_message:
    if time.time() - st.session_state.feedback_time < 10:  # Show for 10 seconds
        st.info(st.session_state.feedback_message)
        feedback_displayed = True
    else:
        st.session_state.feedback_message = None
        st.session_state.feedback_time = 0
        st.session_state.pending_save = False

# Chat input (moved higher and made bigger)
st.markdown("---")

# Custom CSS to make the text area look more like a chat input
st.markdown("""
<style>
.stTextArea > div > div > textarea {
    border-radius: 20px;
    border: 2px solid #e0e0e0;
    padding: 12px 16px;
    font-size: 16px;
    background-color: #f8f9fa;
}
.stTextArea > div > div > textarea:focus {
    border-color: #ff4b4b;
    box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.2);
}
</style>
""", unsafe_allow_html=True)

# Use a form to automatically clear the input after submit
with st.form("message_form", clear_on_submit=True):
    user_input = st.text_area(
        "Type your requirement or command here:",
        placeholder="e.g., 'The system must handle 1000 concurrent users'",
        height=80,
        key="main_input_area",
        label_visibility="collapsed"
    )
    
    # Send button under the text input
    submitted = st.form_submit_button("📤 Send", type="primary", use_container_width=True)
    
    # Handle form submission
    if submitted and user_input.strip():
        st.session_state.user_input_box = user_input.strip()
        process_user_input()

# Analysis report display
if st.session_state.analysis_report:
    st.divider()
    st.subheader("📊 Analysis Report")
    
    # Add timestamp and metadata
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_reqs = sum(len(reqs) for reqs in st.session_state.state["requirements"].values())
    
    # Report metadata bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Requirements Analyzed", total_reqs)
    with col2:
        st.metric("Product", selected_product)
    with col3:
        st.metric("Generated", timestamp.split()[1])  # Just time
    with col4:
        # Report length
        word_count = len(st.session_state.analysis_report.split())
        st.metric("Report Length", f"{word_count} words")
    
    # Action buttons
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        # Enhanced download with timestamp
        filename = f"requirements_analysis_{selected_product.lower().replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        st.download_button(
            label="📥 Download Report",
            data=st.session_state.analysis_report,
            file_name=filename,
            mime="text/markdown",
            type="primary"
        )
    with col2:
        if st.button("🔄 Re-analyze", help="Run analysis again with current requirements"):
            analyze_requirements()
    with col3:
        if st.button("📋 Copy to Clipboard", help="Copy report to clipboard"):
            # Use JavaScript to copy to clipboard
            # Pre-process the string to avoid f-string backslash issues
            escaped_report = st.session_state.analysis_report.replace('`', '\\`')
            st.components.v1.html(f"""
                <script>
                navigator.clipboard.writeText(`{escaped_report}`);
                </script>
            """, height=0)
            st.toast("📋 Report copied to clipboard!")
    with col4:
        if st.button("🗑️ Clear Report", help="Remove this analysis report"):
            st.session_state.analysis_report = None
            st.rerun()
    
    st.markdown("---")
    
    # Display report with expandable sections
    with st.container():
        # Add a search box for large reports
        if word_count > 500:
            search_term = st.text_input("🔍 Search in report:", placeholder="e.g., 'feasibility', 'risk', 'security'")
            if search_term:
                # Highlight search terms (simple implementation)
                highlighted_report = st.session_state.analysis_report.replace(
                    search_term, f"**{search_term}**"
                )
                st.markdown(highlighted_report)
            else:
                st.markdown(st.session_state.analysis_report)
        else:
            st.markdown(st.session_state.analysis_report)

# Footer
st.divider()
st.caption(f"💡 **Tip:** Be specific with requirements! Instead of 'fast system', try 'system responds within 2 seconds under normal load'") 