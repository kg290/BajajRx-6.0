#!/bin/bash

# BajajRx-6.0 Production Startup Script
# Author: kg290
# Date: 2025-07-28 15:46:06
# Optimized for Render deployment with webhook support

echo "[kg290] Starting BajajRx-6.0 Insurance Policy Assistant..."
echo "[kg290] Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S') UTC"

# Set production environment variables
export PYTHONPATH="${PYTHONPATH}:."
export PYTHONUNBUFFERED=1
export ENVIRONMENT=production

# Enhanced logging for hackathon evaluation
echo "[kg290] Environment: $ENVIRONMENT"
echo "[kg290] Python Path: $PYTHONPATH"
echo "[kg290] Working Directory: $(pwd)"

# Display current models being used
echo "[kg290] Model Configuration:"
echo "  - Scout Model: meta-llama/llama-4-scout-17b-16e-instruct (128K context)"
echo "  - Deep Model: llama3-70b-8192"
echo "  - Embedding Model: intfloat/e5-base-v2"

# Verify critical dependencies for the dual LLM system
echo "[kg290] Verifying dependencies..."
python -c "
import fastapi, uvicorn, sentence_transformers, faiss, fitz, requests, numpy
from dotenv import load_dotenv
import json, tempfile, os, asyncio
print('[kg290] All core dependencies verified')
print('[kg290] FastAPI version:', fastapi.__version__)
print('[kg290] sentence-transformers version:', sentence_transformers.__version__)
" || {
    echo "[kg290] ERROR: Missing critical dependencies"
    exit 1
}

# Verify GROQ API key for dual LLM system
if [ -z "$GROQ_API_KEY" ]; then
    echo "[kg290] WARNING: GROQ_API_KEY not set - Llama 4 Scout and Llama 3.1 70B calls will fail"
    echo "[kg290] Please set GROQ_API_KEY in Render environment variables"
else
    echo "[kg290] GROQ_API_KEY configured for dual LLM system"
fi

# Verify model initialization (critical for hackathon performance)
echo "[kg290] Pre-loading sentence transformer model..."
python -c "
from sentence_transformers import SentenceTransformer
import time
start_time = time.time()
model = SentenceTransformer('intfloat/e5-base-v2')
load_time = time.time() - start_time
print(f'[kg290] Model loaded successfully in {load_time:.2f} seconds')
print(f'[kg290] Embedding dimension: {model.get_sentence_embedding_dimension()}')
" || {
    echo "[kg290] WARNING: Model pre-loading failed - will load on first request"
}

# Health check function for hackathon evaluation
health_check() {
    echo "[kg290] Performing startup health check..."
    timeout 15s python -c "
import requests
import time
import json

# Wait for server to start
time.sleep(3)

try:
    # Test health endpoint
    health_response = requests.get('http://localhost:${PORT:-8000}/health', timeout=10)
    if health_response.status_code == 200:
        health_data = health_response.json()
        print(f'[kg290] Health check passed: {health_data.get(\"status\", \"unknown\")}')
        print(f'[kg290] Scout model: {health_data.get(\"scout_model\", \"unknown\")}')
        print(f'[kg290] Deep model: {health_data.get(\"deep_model\", \"unknown\")}')
        exit(0)
    else:
        print(f'[kg290] Health check failed: HTTP {health_response.status_code}')
        exit(1)
except Exception as e:
    print(f'[kg290] Health check error: {e}')
    exit(1)
" &
}

# Start background health check
health_check

# Display hackathon endpoint information
echo "[kg290] Hackathon Endpoint Configuration:"
echo "  - Main endpoint: /hackrx/run (POST)"
echo "  - Development endpoint: /ask (POST)"
echo "  - Health check: /health (GET)"
echo "  - Root info: / (GET)"

# Start the application with optimized settings for hackathon evaluation
echo "[kg290] Starting FastAPI server with dual LLM optimizations..."
echo "[kg290] Optimized for:"
echo "  - Accuracy: Dual LLM routing (Scout → Deep)"
echo "  - Token efficiency: Intelligent routing based on query complexity"
echo "  - Explainability: Detailed reasoning and confidence scores"
echo "  - Latency: Optimized chunking and retrieval"
echo "  - Reusability: Modular architecture with factory functions"

# Optimized uvicorn settings for Render and hackathon performance
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-log \
    --log-level info \
    --timeout-keep-alive 30 \
    --timeout-graceful-shutdown 15 \
    --loop uvloop \
    --http httptools \
    --no-server-header \
    --date-header \
    --ws websockets