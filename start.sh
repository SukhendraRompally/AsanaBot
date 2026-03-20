#!/bin/bash
# Asana Agent — Moveworks Reference Implementation
# Starts the FastAPI server on port 8001.
# Port 8001 avoids conflict with other services on this VM (Typeface on 8000).

set -e
cd "$(dirname "$0")"

echo "Starting AsanaBot on http://0.0.0.0:8001"
echo "  Docs: http://localhost:8001/docs"
echo "  Health: http://localhost:8001/health"
echo "  Tools: http://localhost:8001/tools"

uvicorn app:app --host 0.0.0.0 --port 8001 --reload
