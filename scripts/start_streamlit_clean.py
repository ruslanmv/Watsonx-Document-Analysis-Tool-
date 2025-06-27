#!/usr/bin/env python3
"""
Clean Streamlit startup wrapper to suppress torch warnings.
This imports torch early and suppresses warnings before Streamlit does its module inspection.
"""

import warnings
import sys
import os

# Suppress ALL warnings early
warnings.filterwarnings("ignore")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Early torch import with warning suppression
try:
    import torch
    # Suppress torch-specific warnings
    torch.set_warn_always(False)
except ImportError:
    pass  # torch might not be installed
except Exception:
    pass  # ignore any torch-related errors

# Now start Streamlit with suppressed warnings
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    # Set Streamlit arguments
    sys.argv = [
        "streamlit",
        "run", 
        "frontend/streamlit_app.py",
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
        "--logger.level", "warning",
        "--runner.fastReruns", "true"
    ]
    
    # Start Streamlit
    stcli.main() 