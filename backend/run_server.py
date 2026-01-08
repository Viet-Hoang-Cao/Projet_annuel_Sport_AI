#!/usr/bin/env python3
"""
Startup script for the Exercise AI web application.
Run this script to start the FastAPI server.
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print("=" * 60)
    print("🏋️  Exercise AI - Fitness Form Analyzer")
    print("=" * 60)
    print(f"Starting server on http://{host}:{port}")
    print(f"Open your browser and navigate to: http://localhost:{port}")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )