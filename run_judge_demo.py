#!/usr/bin/env python3
"""
🏆 JUDGE DEMO LAUNCHER
One-click start for the requirements analysis demo

This script:
1. Starts the FastAPI backend server
2. Launches the judge demo interface
3. Provides fallback mock mode if backend fails
"""

import subprocess
import sys
import time
import webbrowser
import signal
import os
from pathlib import Path

def start_api_server():
    """Start the FastAPI backend server directly"""
    try:
        print("🚀 Starting API server...")
        process = subprocess.Popen([
            sys.executable, "api_server.py"
        ])
        
        # Give it a moment to start
        time.sleep(3)
        
        # Check if it's running
        if process.poll() is None:
            print("✅ API server started successfully")
            return process
        else:
            print("❌ API server failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def start_demo_interface():
    """Start the Streamlit demo interface"""
    try:
        print("🎬 Starting judge demo interface...")
        
        # Use poetry run streamlit command
        process = subprocess.Popen([
            "poetry", "run", "streamlit", "run", "judge_demo.py", 
            "--server.port", "8502", 
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
        # Wait a moment then open browser
        time.sleep(2)
        print("🌐 Opening demo in browser...")
        webbrowser.open("http://localhost:8502")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start demo interface: {e}")
        return None

def cleanup_processes(api_process, demo_process):
    """Clean up processes properly"""
    print("\n🛑 Stopping demo...")
    
    # Stop demo interface
    if demo_process:
        print("   Stopping demo interface...")
        demo_process.terminate()
        time.sleep(1)
        if demo_process.poll() is None:
            demo_process.kill()
    
    # Stop API server
    if api_process:
        print("   Stopping API server...")
        api_process.terminate()
        time.sleep(1)
        if api_process.poll() is None:
            api_process.kill()
    
    print("✅ Demo stopped")

def main():
    print("=" * 60)
    print("🏆 JUDGE DEMO LAUNCHER")
    print("Requirements Analysis Tool - AI-Powered Evidence-Based Analysis")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("judge_demo.py").exists():
        print("❌ judge_demo.py not found. Please run this from the project root.")
        sys.exit(1)
    
    if not Path("api_server.py").exists():
        print("❌ api_server.py not found. Please run this from the project root.")
        sys.exit(1)
    
    # Check if poetry is available
    try:
        subprocess.run(["poetry", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Poetry not found. Please install Poetry or use manual launch:")
        print("   streamlit run judge_demo.py --server.port 8502")
        sys.exit(1)
    
    print("\n📋 Starting demo components...")
    
    api_process = None
    demo_process = None
    
    try:
        # Start API server (optional - demo can run in mock mode)
        api_process = start_api_server()
        
        # Start demo interface
        demo_process = start_demo_interface()
        
        if demo_process is None:
            print("❌ Failed to start demo. Please check the setup.")
            if api_process:
                api_process.terminate()
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("🎉 DEMO READY!")
        print("=" * 60)
        print("📍 Demo URL: http://localhost:8502")
        
        if api_process:
            print("🔗 API Server: http://localhost:8000 (running)")
            print("💡 Full analysis features available")
        else:
            print("⚠️  API Server: Not running")
            print("💡 Demo will use mock mode for analysis results")
        
        print("\n🎯 Instructions for Judges:")
        print("1. Click 'Start Demo' to begin")
        print("2. Select number of requirements (3-5)")
        print("3. Review auto-selected diverse requirements")
        print("4. Run AI analysis to see the complete pipeline")
        print("5. Review detailed analysis report")
        
        print("\n⏹️  Press Ctrl+C to stop the demo")
        print("=" * 60)
        
        # Simple wait loop
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
            
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_processes(api_process, demo_process)

if __name__ == "__main__":
    main() 