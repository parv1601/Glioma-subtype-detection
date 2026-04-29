#!/usr/bin/env python3
"""
Interactive WSI Heatmap Viewer Launcher
Activates virtual environment and starts Streamlit app
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("Starting Interactive WSI Heatmap Viewer...")
    print()
    
    # Get project directory
    project_dir = Path(__file__).parent.resolve()
    os.chdir(project_dir)
    
    # Check virtual environment
    venv_activate = project_dir / "venv" / "bin" / "activate"
    if not venv_activate.exists():
        print("Virtual environment not found!")
        print(f"   Expected: {venv_activate}")
        print("   Please create it with: python -m venv venv")
        sys.exit(1)
    
    print("Virtual environment found")
    print()
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("Streamlit is installed")
    except ImportError:
        print("Streamlit not found. Installing...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "streamlit"],
            check=True
        )
    
    print()
    print("Starting Streamlit application...")
    print("Open your browser to: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    
    # Start streamlit app
    viewer_script = project_dir / "interactive_heatmap_viewer.py"
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(viewer_script)],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\nStreamlit server stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
