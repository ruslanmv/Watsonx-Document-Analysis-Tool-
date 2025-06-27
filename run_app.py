#!/usr/bin/env python3
"""
Startup script for the Requirements Bot application.
Can run the FastAPI server, Streamlit app, or both.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def run_fastapi():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI Analysis Server...")
    return subprocess.Popen([
        sys.executable, "api_server.py"
    ], cwd=Path(__file__).parent)

def run_streamlit():
    """Start the Streamlit app"""
    print("🚀 Starting Streamlit Frontend...")
    
    # Set environment variables to suppress warnings before Streamlit starts
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["STREAMLIT_LOGGER_LEVEL"] = "warning"
    env["TORCH_LOGS"] = "+none"
    
    return subprocess.Popen([
        sys.executable, "scripts/start_streamlit_clean.py"
    ], cwd=Path(__file__).parent, env=env)

def main():
    parser = argparse.ArgumentParser(description="Requirements Bot Launcher")
    parser.add_argument(
        "mode", 
        nargs='?',  # Make it optional
        default="both",  # Set default
        choices=["api", "frontend", "both"], 
        help="What to run: 'api' (FastAPI server), 'frontend' (Streamlit), or 'both' (default: both)"
    )
    parser.add_argument(
        "--no-browser", 
        action="store_true", 
        help="Don't open browser automatically"
    )
    
    args = parser.parse_args()
    
    processes = []
    
    try:
        if args.mode in ["api", "both"]:
            api_process = run_fastapi()
            processes.append(("FastAPI", api_process))
            
            # Wait a moment for API to start
            if args.mode == "both":
                print("⏳ Waiting for API server to start...")
                time.sleep(3)
        
        if args.mode in ["frontend", "both"]:
            streamlit_process = run_streamlit()
            processes.append(("Streamlit", streamlit_process))
            
            # Open browser unless disabled
            if not args.no_browser:
                time.sleep(2)
                try:
                    import webbrowser
                    webbrowser.open("http://localhost:8501")
                except:
                    pass
        
        print("\n✅ Application started successfully!")
        if args.mode == "both":
            print("📊 API Server: http://localhost:8000")
            print("🖥️  Frontend:   http://localhost:8501")
            print("📚 API Docs:   http://localhost:8000/docs")
        elif args.mode == "api":
            print("📊 API Server: http://localhost:8000")
            print("📚 API Docs:   http://localhost:8000/docs")
        else:
            print("🖥️  Frontend: http://localhost:8501")
            print("⚠️  Note: Analysis features require the API server to be running separately")
        
        print("\n💡 Press Ctrl+C to stop")
        
        # Wait for processes
        while processes:
            for name, process in processes[:]:
                if process.poll() is not None:
                    print(f"❌ {name} process stopped")
                    processes.remove((name, process))
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        for name, process in processes:
            print(f"   Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("✅ Shutdown complete")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("frontend/streamlit_app.py"):
        print("❌ Error: Please run this script from the project root directory")
        print("   (where frontend/streamlit_app.py exists)")
        sys.exit(1)
    
    main() 