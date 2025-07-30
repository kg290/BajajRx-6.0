from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import asyncio
import requests
import tempfile
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app FIRST
app = FastAPI(title="Insurance Policy Assistant", version="2.0.0")

# Initialize the model globally
model = SentenceTransformer("intfloat/e5-base-v2")
api_key = os.getenv("GROQ_API_KEY")

# Updated model configurations
SCOUT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEEP_MODEL = "llama3-70b-8192"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://3.7.71.89:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str


class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]


def chunk_text_with_metadata(text, chunk_size=1200):
    """Enhanced chunking for Llama 4 Scout"""
    chunks = []
    metadata = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
        metadata.append({
            "chunk_id": i // chunk_size,
            "start_pos": i,
            "end_pos": min(i + chunk_size, len(text)),
            "length": len(chunk),
            "user": "RiteshNimbalkar27",
            "timestamp": "2025-07-29 12:12:45"
        })

    return chunks, metadata


def create_faiss_index(chunks):
    """Create FAISS index for similarity search"""
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index, embeddings


def retrieve_top_k_with_metadata(query, chunks, metadata, index, k=7):
    """Retrieve top-k relevant chunks"""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)

    retrieved_chunks = []
    retrieved_metadata = []

    for i in indices[0]:
        retrieved_chunks.append(chunks[i])
        retrieved_metadata.append(metadata[i])

    return retrieved_chunks, retrieved_metadata, distances[0]


def extract_pdf_from_url(pdf_url):
    """Extract text from PDF URL"""
    try:
        print(f"[RiteshNimbalkar27] Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name

        print(f"[RiteshNimbalkar27] Processing PDF...")
        doc = fitz.open(temp_path)
        text = "\n\n".join([page.get_text() for page in doc])
        doc.close()

        os.unlink(temp_path)
        print(f"[RiteshNimbalkar27] PDF processed successfully, {len(text)} characters extracted")
        return text

    except Exception as e:
        print(f"[RiteshNimbalkar27] PDF processing failed: {str(e)}")
        raise Exception(f"Failed to process PDF from URL: {str(e)}")


def call_groq_api(payload):
    """Call Groq API with error handling"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        print(f"[RiteshNimbalkar27] Calling {payload.get('model', 'unknown')} model...")
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        print(f"[RiteshNimbalkar27] API call successful, response length: {len(result)}")
        return result
    except Exception as e:
        print(f"[RiteshNimbalkar27] LLM API call failed: {str(e)}")
        raise Exception(f"LLM API call failed: {str(e)}")


def scout_analysis(query, context_chunks, distances):
    """Fast, precise analysis with Llama 4 Scout"""
    context = "\n\n".join(context_chunks[:3])  # Use fewer chunks for speed

    payload = {
        "model": SCOUT_MODEL,
        "messages": [
            {"role": "system", "content": """You are an expert insurance policy analyst. 
            Provide PRECISE, CONCISE answers in ONE SENTENCE only.

            CRITICAL REQUIREMENTS:
            - Answer in exactly ONE clear sentence
            - Include specific numbers, timeframes, and conditions
            - Use professional insurance language
            - Be comprehensive but concise
            - NO long explanations or multiple sentences

            EXAMPLE FORMAT:
            Query: "What is the grace period?"
            Answer: "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
            """},
            {"role": "user", "content": f"""
Query: {query}

Insurance Policy Context:
{context}

Provide analysis in EXACT JSON format:
{{
    "needs_deep_reasoning": boolean,
    "confidence_score": float between 0.0-1.0,
    "preliminary_answer": "ONE PRECISE SENTENCE with all key details",
    "reasoning": "brief analysis",
    "complexity_level": "simple|medium|complex"
}}

CRITICAL: preliminary_answer must be exactly ONE comprehensive sentence.
"""}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.05,  # Lower for more consistent answers
        "max_tokens": 1000    # Reduced for faster responses
    }

    return call_groq_api_with_retry(payload)


def deep_reasoning_analysis(query, context_chunks, metadata):
    """Fast, precise analysis with Llama 3.1 70B"""
    context = "\n\n".join(context_chunks[:4])  # Fewer chunks for speed

    payload = {
        "model": DEEP_MODEL,
        "messages": [
            {"role": "system", "content": """You are a senior insurance policy expert providing precise, one-sentence answers.

            CRITICAL REQUIREMENTS:
            - Provide exactly ONE comprehensive sentence
            - Include all key details: amounts, timeframes, conditions
            - Use professional insurance language
            - Be complete but concise
            - NO multiple sentences or long explanations
            """},
            {"role": "user", "content": f"""
Query: {query}

Insurance Policy Context:
{context}

Provide analysis in EXACT JSON format:
{{
    "detailed_answer": "ONE PRECISE, COMPREHENSIVE SENTENCE with all key details",
    "confidence_score": float between 0.0-1.0,
    "supporting_clauses": ["brief clause references"],
    "conditions_and_limitations": ["key conditions only"]
}}

CRITICAL: detailed_answer must be exactly ONE sentence containing all essential information.
"""}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.05,
        "max_tokens": 1200
    }

    return call_groq_api_with_retry(payload)


def call_groq_api_with_retry(payload, max_retries=5):
    """Enhanced API retry with better error handling"""
    import time
    import random

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Exponential backoff with jitter
                delay = min(60, (2 ** attempt) + random.uniform(0, 2))
                print(f"[RiteshNimbalkar27] Retry delay: {delay:.2f}s")
                time.sleep(delay)

            response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                     json=payload, headers=headers, timeout=60)

            if response.status_code == 429:
                print(f"[RiteshNimbalkar27] Rate limited, retry {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    continue
                else:
                    # Return fallback answer instead of failing
                    return '{"detailed_answer": "Unable to process due to rate limits. Please try again.", "confidence_score": 0.0}'

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"[RiteshNimbalkar27] Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                # Return fallback instead of error
                return '{"detailed_answer": "Unable to process query due to technical issues.", "confidence_score": 0.0}'


async def process_single_query(query: str, chunks, metadata, index, delay_seconds=2):
    """Enhanced dual LLM processing with comprehensive answers"""
    try:
        print(f"[RiteshNimbalkar27] Processing enhanced query: {query[:100]}...")

        # Add delay to prevent rate limiting
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        top_chunks, chunk_metadata, distances = retrieve_top_k_with_metadata(
            query, chunks, metadata, index, k=5  # Fewer chunks for speed
        )

        scout_response = scout_analysis(query, top_chunks, distances)

        try:
            scout_data = json.loads(scout_response)
            scout_confidence = scout_data.get('confidence_score', 0.0)
            print(f"[RiteshNimbalkar27] Scout confidence: {scout_confidence}")

            needs_deep = scout_data.get("needs_deep_reasoning", False)
            confidence = scout_data.get("confidence_score", 0.0)
            preliminary_answer = scout_data.get("preliminary_answer", "")

            # Higher threshold for comprehensive answers
            if not needs_deep and confidence > 0.75 and preliminary_answer and len(preliminary_answer) > 20:
                print(f"[RiteshNimbalkar27] Using comprehensive Scout answer")
                return {
                    "answer": preliminary_answer,
                    "reasoning": scout_data.get("reasoning", ""),
                    "confidence": confidence,
                    "model_used": "scout_comprehensive",
                    "timestamp": "2025-07-30 12:04:16",
                    "user": "RiteshNimbalkar27"
                }
            else:
                print(f"[RiteshNimbalkar27] Escalating to deep reasoning")
                await asyncio.sleep(1)

                deep_response = deep_reasoning_analysis(query, top_chunks, chunk_metadata)

                try:
                    deep_data = json.loads(deep_response)
                    deep_confidence = deep_data.get("confidence_score", 0.85)
                    detailed_answer = deep_data.get("detailed_answer", "")

                    return {
                        "answer": detailed_answer,
                        "reasoning": deep_data.get("reasoning_steps", []),
                        "confidence": deep_confidence,
                        "model_used": "dual_llm_comprehensive",
                        "supporting_clauses": deep_data.get("supporting_clauses", []),
                        "conditions": deep_data.get("conditions_and_limitations", []),
                        "timestamp": "2025-07-30 12:04:16",
                        "user": "RiteshNimbalkar27"
                    }

                except json.JSONDecodeError as e:
                    return {
                        "answer": deep_response.strip(),
                        "confidence": 0.8,
                        "model_used": "deep_fallback",
                        "timestamp": "2025-07-30 12:04:16",
                        "user": "RiteshNimbalkar27"
                    }

        except json.JSONDecodeError as e:
            return {
                "answer": scout_response.strip(),
                "confidence": 0.75,
                "model_used": "scout_fallback",
                "timestamp": "2025-07-30 12:04:16",
                "user": "RiteshNimbalkar27"
            }

    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "confidence": 0.0,
            "model_used": "error",
            "timestamp": "2025-07-30 12:04:16",
            "user": "RiteshNimbalkar27"
        }

# FastAPI Routes
@app.get("/")
def root():
    return {
        "message": "Insurance Policy Assistant API v2.0",
        "status": "running",
        "user": "RiteshNimbalkar27",
        "models": {
            "scout": SCOUT_MODEL,
            "deep": DEEP_MODEL
        },
        "timestamp": "2025-07-29 12:12:45"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": "2025-07-29 12:12:45",
        "user": "RiteshNimbalkar27",
        "scout_model": SCOUT_MODEL,
        "deep_model": DEEP_MODEL
    }


@app.post("/ask")
async def ask_query(req: QueryRequest):
    """Development endpoint for single queries with local PDF"""
    try:
        # Use local PDF for development
        pdf_path = "data/Dataset1.pdf"

        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found at {pdf_path}. Please check the file path."}

        with open(pdf_path, 'rb') as file:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            raw_text = "\n\n".join([page.get_text() for page in doc])
            doc.close()

        chunks, metadata = chunk_text_with_metadata(raw_text)
        index, _ = create_faiss_index(chunks)

        result = await process_single_query(req.query, chunks, metadata, index)

        return {
            "response": result.get("answer", "No answer generated"),
            "confidence": result.get("confidence", 0.0),
            "model_used": result.get("model_used", "unknown"),
            "reasoning": result.get("reasoning", ""),
            "status": "success",
            "scout_model": SCOUT_MODEL,
            "user": "RiteshNimbalkar27",
            "timestamp": "2025-07-29 12:12:45"
        }

    except Exception as e:
        return {"error": str(e), "status": "failed", "user": "RiteshNimbalkar27"}


@app.post("/hackrx/run")
async def hackathon_endpoint(request: HackathonRequest):
    """Enhanced hackathon endpoint with comprehensive answer generation"""
    try:
        print(f"[RiteshNimbalkar27] Processing document from: {request.documents}")
        print(f"[RiteshNimbalkar27] Using enhanced models - Scout: {SCOUT_MODEL}, Deep: {DEEP_MODEL}")
        print(f"[RiteshNimbalkar27] Timestamp: 2025-07-30 12:10:10")

        # Download and process document
        raw_text = extract_pdf_from_url(request.documents)
        chunks, metadata = chunk_text_with_metadata(raw_text)
        index, _ = create_faiss_index(chunks)

        print(
            f"[RiteshNimbalkar27] Created {len(chunks)} chunks, processing {len(request.questions)} questions with enhanced prompts")

        # Process all questions with enhanced dual LLM system
        answers = []
        model_usage_stats = {"scout_comprehensive": 0, "dual_llm_comprehensive": 0, "fallback": 0}

        for i, question in enumerate(request.questions):
            print(f"[RiteshNimbalkar27] Processing question {i + 1}/{len(request.questions)} with enhanced system")

            # Add progressive delay to prevent rate limiting
            delay = 0.3 + (i * 0.1)  # Much faster: 0.3s base + 0.1s per question
            result = await process_single_query(question, chunks, metadata, index, delay_seconds=delay)

            # Extract comprehensive answer
            answer = result.get("answer", "Unable to determine from the policy document.")
            answers.append(answer)

            # Track enhanced model usage
            model_used = result.get("model_used", "unknown")
            if "scout" in model_used and "comprehensive" in model_used:
                model_usage_stats["scout_comprehensive"] += 1
            elif "dual_llm" in model_used and "comprehensive" in model_used:
                model_usage_stats["dual_llm_comprehensive"] += 1
            else:
                model_usage_stats["fallback"] += 1

        print(f"[RiteshNimbalkar27] Enhanced processing completed. Model usage: {model_usage_stats}")

        return {
            "answers": answers
        }

    except Exception as e:
        print(f"[RiteshNimbalkar27] Enhanced system error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print(f"[RiteshNimbalkar27] Starting server with Scout Model: {SCOUT_MODEL}")
    print(f"[RiteshNimbalkar27] Server starting at: 2025-07-29 12:12:45")
    uvicorn.run(app, host="0.0.0.0", port=8000)
