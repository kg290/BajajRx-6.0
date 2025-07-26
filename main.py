from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="Insurance Policy Assistant", version="2.0.0")

# Initialize the model globally
model = SentenceTransformer("intfloat/e5-base-v2")
api_key = os.getenv("GROQ_API_KEY")

# Updated model configurations
SCOUT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Updated from llama3-8b-8192
DEEP_MODEL = "llama3-70b-8192"


class QueryRequest(BaseModel):
    query: str


class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]


def chunk_text_with_metadata(text, chunk_size=1200):  # Increased chunk size for Llama 4
    chunks = []
    metadata = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
        metadata.append({
            "chunk_id": i // chunk_size,
            "start_pos": i,
            "end_pos": min(i + chunk_size, len(text)),
            "length": len(chunk)
        })

    return chunks, metadata


def create_faiss_index(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index, embeddings


def retrieve_top_k_with_metadata(query, chunks, metadata, index, k=7):  # Increased k for better context
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)

    retrieved_chunks = []
    retrieved_metadata = []

    for i in indices[0]:
        retrieved_chunks.append(chunks[i])
        retrieved_metadata.append(metadata[i])

    return retrieved_chunks, retrieved_metadata, distances[0]


def extract_pdf_from_url(pdf_url):
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name

        doc = fitz.open(temp_path)
        text = "\n\n".join([page.get_text() for page in doc])
        doc.close()

        os.unlink(temp_path)
        return text

    except Exception as e:
        raise Exception(f"Failed to process PDF from URL: {str(e)}")


def call_groq_api(payload):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise Exception(f"LLM API call failed: {str(e)}")


def scout_analysis(query, context_chunks, distances):
    """Fast analysis with Llama 4 Scout - Updated from llama3-8b-8192"""

    # Use more context due to 128K context window
    context = "\n\n".join(context_chunks[:5])  # Use top 5 chunks

    payload = {
        "model": SCOUT_MODEL,  # meta-llama/llama-4-scout-17b-16e-instruct
        "messages": [
            {"role": "system", "content": """You are an expert insurance policy analyst with Llama 4 Scout capabilities. 
            Analyze insurance queries with high precision and determine if deep reasoning is needed.

            Provide structured JSON responses with guaranteed formatting."""},
            {"role": "user", "content": f"""
Query: {query}

Policy Context:
{context}

Confidence Scores: {distances.tolist()[:5]}

Analyze this insurance query and respond with EXACT JSON format:
{{
    "needs_deep_reasoning": boolean,
    "confidence_score": float between 0.0-1.0,
    "preliminary_answer": "detailed answer if confident enough",
    "reasoning": "step-by-step analysis",
    "complexity_level": "simple|medium|complex",
    "relevant_clauses": ["clause1", "clause2"]
}}

Rules:
- If confidence_score > 0.85 and complexity_level is "simple", provide preliminary_answer
- If query involves multiple conditions, calculations, or legal interpretations, set needs_deep_reasoning: true
- Always provide reasoning for your decision
"""}
        ],
        "response_format": {"type": "json_object"},  # Guaranteed JSON with Llama 4 Scout
        "temperature": 0.1,
        "max_tokens": 2000
    }

    return call_groq_api(payload)


def deep_reasoning_analysis(query, context_chunks, metadata):
    """Comprehensive analysis with Llama 3.1 70B - Enhanced logging version"""

    context = "\n\n".join(context_chunks)

    print(f"[kg290] Starting deep reasoning analysis with Llama 3.1 70B...")
    print(f"[kg290] Context length: {len(context)} characters, {len(context_chunks)} chunks")
    print(f"[kg290] Timestamp: 2025-07-26 11:54:03")

    payload = {
        "model": DEEP_MODEL,  # llama3-70b-8192
        "messages": [
            {"role": "system", "content": """You are a senior insurance policy expert providing comprehensive analysis.
            Provide detailed, accurate answers with clear reasoning and clause references.

            Always respond in valid JSON format with high confidence scores for thorough analysis."""},
            {"role": "user", "content": f"""
Query: {query}

Complete Policy Context:
{context}

Metadata: {json.dumps(metadata, indent=2)}

Provide a comprehensive analysis in EXACT JSON format:
{{
    "detailed_answer": "comprehensive answer with specific details",
    "supporting_clauses": ["exact clause text references"],
    "conditions_and_limitations": ["list of relevant conditions"],
    "confidence_score": float between 0.0-1.0,
    "reasoning_steps": ["step1", "step2", "step3"],
    "additional_considerations": "any important notes or exceptions",
    "analysis_depth": "comprehensive|detailed|standard",
    "certainty_level": "high|medium|low",
    "processing_timestamp": "2025-07-26 11:54:03",
    "processed_by": "kg290"
}}

Requirements:
- Reference specific policy clauses with exact clause numbers
- Include all relevant conditions and limitations
- Provide step-by-step reasoning chain
- Maintain high accuracy and precision standards
- Set confidence_score based on completeness of analysis (aim for 0.85+ for comprehensive responses)
- Include specific policy section references
- Explain complex insurance terms clearly
"""}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 3000
    }

    print(f"[kg290] Calling Llama 3.1 70B API for deep analysis...")

    try:
        response = call_groq_api(payload)
        print(f"[kg290] Deep model response received successfully")
        print(f"[kg290] Response length: {len(response)} characters")

        # Try to parse and validate the response
        try:
            parsed_response = json.loads(response)
            confidence = parsed_response.get("confidence_score", "unknown")
            analysis_depth = parsed_response.get("analysis_depth", "unknown")
            print(f"[kg290] Deep analysis confidence: {confidence}")
            print(f"[kg290] Analysis depth: {analysis_depth}")
            print(f"[kg290] Deep reasoning analysis completed at 2025-07-26 11:54:03")

        except json.JSONDecodeError:
            print(f"[kg290] Warning: Deep model response is not valid JSON, but proceeding...")

        return response

    except Exception as e:
        print(f"[kg290] Error in deep reasoning analysis: {str(e)}")
        print(f"[kg290] Error timestamp: 2025-07-26 11:54:03")

        # Return a fallback response
        fallback_response = {
            "detailed_answer": f"Deep analysis failed due to API error: {str(e)}. Please try again.",
            "supporting_clauses": [],
            "conditions_and_limitations": ["Analysis incomplete due to technical error"],
            "confidence_score": 0.0,
            "reasoning_steps": ["Deep analysis API call failed", "Returning fallback response"],
            "additional_considerations": "Technical error occurred during analysis",
            "analysis_depth": "failed",
            "certainty_level": "low",
            "processing_timestamp": "2025-07-26 11:54:03",
            "processed_by": "kg290",
            "error": str(e)
        }

        return json.dumps(fallback_response)


async def process_single_query(query: str, chunks, metadata, index):
    """Updated dual LLM processing with enhanced confidence logging"""
    try:
        print(f"[kg290] Processing query with Llama 4 Scout: {query[:100]}...")
        print(f"[kg290] Query processing started at: 2025-07-26 11:54:03")

        # Retrieve relevant chunks
        top_chunks, chunk_metadata, distances = retrieve_top_k_with_metadata(
            query, chunks, metadata, index, k=7
        )

        # Scout analysis with Llama 4 Scout
        scout_response = scout_analysis(query, top_chunks, distances)

        try:
            scout_data = json.loads(scout_response)
            scout_confidence = scout_data.get('confidence_score', 0.0)
            print(f"[kg290] Scout analysis complete. Confidence: {scout_confidence}")

            # Decision logic
            needs_deep = scout_data.get("needs_deep_reasoning", False)
            confidence = scout_data.get("confidence_score", 0.0)
            preliminary_answer = scout_data.get("preliminary_answer", "")

            if not needs_deep and confidence > 0.85 and preliminary_answer:
                print(f"[kg290] Using Scout preliminary answer (high confidence: {confidence})")
                print(f"[kg290] Scout-only response completed at: 2025-07-26 11:54:03")
                return {
                    "answer": preliminary_answer,
                    "reasoning": scout_data.get("reasoning", ""),
                    "confidence": confidence,
                    "model_used": "scout_only",
                    "complexity": scout_data.get("complexity_level", "simple"),
                    "timestamp": "2025-07-26 11:54:03",
                    "user": "kg290"
                }
            else:
                print(f"[kg290] Escalating to deep reasoning model (scout confidence: {confidence})")

                # Deep analysis with Llama 3.1 70B
                deep_response = deep_reasoning_analysis(query, top_chunks, chunk_metadata)

                try:
                    deep_data = json.loads(deep_response)
                    deep_confidence = deep_data.get("confidence_score", 0.8)

                    # Enhanced logging for both models
                    print(f"[kg290] Deep analysis complete. Confidence: {deep_confidence}")
                    print(f"[kg290] Model progression: Scout({confidence:.3f}) → Deep({deep_confidence:.3f})")
                    print(f"[kg290] Dual LLM analysis completed at: 2025-07-26 11:54:03")

                    return {
                        "answer": deep_data.get("detailed_answer", "Unable to provide detailed analysis"),
                        "reasoning": deep_data.get("reasoning_steps", []),
                        "confidence": deep_confidence,  # Use deep model confidence
                        "model_used": "dual_llm",
                        "supporting_clauses": deep_data.get("supporting_clauses", []),
                        "conditions": deep_data.get("conditions_and_limitations", []),
                        "scout_confidence": confidence,  # Include scout confidence for comparison
                        "deep_confidence": deep_confidence,
                        "analysis_depth": deep_data.get("analysis_depth", "comprehensive"),
                        "certainty_level": deep_data.get("certainty_level", "high"),
                        "timestamp": "2025-07-26 11:54:03",
                        "user": "kg290"
                    }
                except json.JSONDecodeError as e:
                    print(f"[kg290] Deep model JSON parsing failed: {str(e)}")
                    print(f"[kg290] Using raw deep response with estimated confidence: 0.8")
                    return {
                        "answer": deep_response.strip(),
                        "reasoning": "Processed with deep reasoning model (JSON parsing failed)",
                        "confidence": 0.8,
                        "model_used": "deep_fallback",
                        "scout_confidence": confidence,
                        "parsing_error": str(e),
                        "timestamp": "2025-07-26 11:54:03",
                        "user": "kg290"
                    }

        except json.JSONDecodeError as e:
            print(f"[kg290] Scout JSON parsing failed: {str(e)}")
            print(f"[kg290] Using direct scout response with estimated confidence: 0.7")
            return {
                "answer": scout_response.strip(),
                "reasoning": "Processed with Llama 4 Scout (JSON parsing failed)",
                "confidence": 0.7,
                "model_used": "scout_fallback",
                "parsing_error": str(e),
                "timestamp": "2025-07-26 11:54:03",
                "user": "kg290"
            }

    except Exception as e:
        print(f"[kg290] Critical error in query processing: {str(e)}")
        print(f"[kg290] Error occurred at: 2025-07-26 11:54:03")
        return {
            "answer": f"Error processing query: {str(e)}",
            "reasoning": "System error occurred during processing",
            "confidence": 0.0,
            "model_used": "error",
            "error_details": str(e),
            "timestamp": "2025-07-26 11:54:03",
            "user": "kg290"
        }
@app.get("/")
def root():
    return {
        "message": "Insurance Policy Assistant API v2.0",
        "status": "running",
        "user": "kg290",
        "models": {
            "scout": SCOUT_MODEL,
            "deep": DEEP_MODEL
        },
        "timestamp": "2025-07-26 11:37:56"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": "2025-07-26 11:37:56",
        "user": "kg290",
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
            "scout_model": SCOUT_MODEL
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.post("/hackrx/run")
async def hackathon_endpoint(request: HackathonRequest):
    """Main hackathon endpoint with updated Llama 4 Scout"""
    try:
        print(f"[kg290] Processing document from: {request.documents}")
        print(f"[kg290] Using models - Scout: {SCOUT_MODEL}, Deep: {DEEP_MODEL}")

        # Download and process document
        raw_text = extract_pdf_from_url(request.documents)
        chunks, metadata = chunk_text_with_metadata(raw_text)
        index, _ = create_faiss_index(chunks)

        print(f"[kg290] Created {len(chunks)} chunks, processing {len(request.questions)} questions")

        # Process all questions with dual LLM system
        answers = []
        model_usage_stats = {"scout_only": 0, "dual_llm": 0, "fallback": 0}

        for i, question in enumerate(request.questions):
            print(f"[kg290] Processing question {i + 1}/{len(request.questions)} with Llama 4 Scout")
            result = await process_single_query(question, chunks, metadata, index)

            # Extract just the answer for the response
            answer = result.get("answer", "Unable to determine from the policy document.")
            answers.append(answer)

            # Track model usage
            model_used = result.get("model_used", "unknown")
            if model_used in model_usage_stats:
                model_usage_stats[model_used] += 1
            else:
                model_usage_stats["fallback"] += 1

        print(f"[kg290] Completed processing. Model usage: {model_usage_stats}")

        return {
            "answers": answers,
            "_metadata": {
                "total_questions": len(request.questions),
                "model_usage": model_usage_stats,
                "scout_model": SCOUT_MODEL,
                "deep_model": DEEP_MODEL,
                "user": "kg290",
                "timestamp": "2025-07-26 11:37:56"
            }
        }

    except Exception as e:
        print(f"[kg290] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print(f"[kg290] Starting server with Scout Model: {SCOUT_MODEL}")
    uvicorn.run(app, host="127.0.0.1", port=8000)