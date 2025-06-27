#!/usr/bin/env python3
"""
Force reset ChromaDB when normal reset fails due to file locking.
Use this when you get PermissionError during --reset.
"""

import sys
import os
import time
import gc
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_chroma_db_path

def force_reset_database():
    """Force reset the ChromaDB with multiple strategies."""
    
    db_path = Path(get_chroma_db_path())
    
    print("🔨 Force Reset ChromaDB")
    print("=" * 40)
    print(f"📂 Database path: {db_path.absolute()}")
    
    if not db_path.exists():
        print("✅ Database directory doesn't exist - nothing to reset")
        return True
    
    print(f"📁 Database directory exists with size: {sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())} bytes")
    
    # Strategy 1: Force garbage collection and wait
    print("\n🔄 Strategy 1: Cleanup and standard removal...")
    gc.collect()  # Force garbage collection
    time.sleep(1)  # Give Windows time to release handles
    
    try:
        shutil.rmtree(db_path)
        print("✅ Success! Database reset with standard method")
        return True
    except PermissionError as e:
        print(f"❌ Standard method failed: {e}")
    
    # Strategy 2: Retry with longer delays
    print("\n🔄 Strategy 2: Retry with delays...")
    for attempt in range(5):
        try:
            time.sleep(2)  # Wait longer between attempts
            shutil.rmtree(db_path)
            print(f"✅ Success! Database reset on attempt {attempt + 1}")
            return True
        except PermissionError:
            print(f"   Attempt {attempt + 1}/5 failed, waiting...")
    
    # Strategy 3: Manual instructions
    print("\n❌ Automatic reset failed. Manual steps required:")
    print(f"   1. Close ALL Python processes and terminals")
    print(f"   2. Open File Explorer and navigate to:")
    print(f"      {db_path.absolute()}")
    print(f"   3. Delete the entire 'chroma_db' folder")
    print(f"   4. Run: python scripts/ingest_docs.py --reset")
    print(f"\n   Alternative: Restart your computer and try again")
    
    return False

def check_running_processes():
    """Check for running Python processes that might be using the database."""
    try:
        import psutil
        python_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'requirement' in cmdline.lower() or 'chroma' in cmdline.lower():
                        python_processes.append((proc.info['pid'], cmdline))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if python_processes:
            print(f"\n⚠️  Found {len(python_processes)} potentially relevant Python processes:")
            for pid, cmdline in python_processes:
                print(f"   PID {pid}: {cmdline[:80]}...")
            print(f"   Consider closing these processes before reset")
        else:
            print(f"✅ No relevant Python processes found running")
            
    except ImportError:
        print(f"💡 Install psutil for process checking: pip install psutil")

def main():
    print("This script will forcefully reset the ChromaDB database.")
    print("Use this when normal 'ingest_docs.py --reset' fails with permission errors.")
    
    # Check for running processes
    check_running_processes()
    
    response = input("\nProceed with force reset? (y/N): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        return
    
    success = force_reset_database()
    
    if success:
        print(f"\n🎉 Database reset complete!")
        print(f"   Now run: python scripts/ingest_docs.py --reset")
    else:
        print(f"\n❌ Automatic reset failed. Please follow manual instructions above.")

if __name__ == "__main__":
    main() 