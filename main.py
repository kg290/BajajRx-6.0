from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
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
import logging
import time

# Import optimization modules
from rate_limiter import create_rate_limiter, RateLimitConfig, generate_context_hash
from response_optimizer import create_response_optimizer, optimize_for_hackathon

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Insurance Policy Assistant", version="3.0.0")

# Initialize optimized components
rate_limit_config = RateLimitConfig(
    requests_per_minute=25,  # Conservative limit for Groq API
    requests_per_hour=800,
    max_retries=3,
    base_delay=2.0,
    max_delay=60.0,
    cache_ttl=1800,  # 30 minutes cache
    enable_caching=True
)

rate_limiter = create_rate_limiter(rate_limit_config)
response_optimizer = create_response_optimizer()

# Initialize the model globally with error handling
try:
    model = SentenceTransformer("intfloat/e5-base-v2")
    logger.info("[kg290] SentenceTransformer model loaded successfully")
except Exception as e:
    logger.warning(f"[kg290] Model loading failed: {e}. Will use fallback if needed.")
    model = None

api_key = os.getenv("GROQ_API_KEY")

# Updated model configurations with optimization
SCOUT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEEP_MODEL = "llama3-70b-8192"

# Global stats tracking
request_stats = {
    "total_requests": 0,
    "cached_responses": 0,
    "rate_limited": 0,
    "errors": 0,
    "optimization_applied": 0
}


class QueryRequest(BaseModel):
    query: str


class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]


class OptimizedResponse(BaseModel):
    answer: str
    confidence: float
    cached: Optional[bool] = False
    optimization_applied: Optional[bool] = False
    processing_time: Optional[float] = None


def chunk_text_with_metadata(text, chunk_size=1200):  # Optimized chunk size
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
    if model is None:
        # Fallback: create dummy index
        logger.warning("[kg290] Using fallback index due to model loading failure")
        dim = 768  # Standard embedding dimension
        embeddings = np.random.rand(len(chunks), dim).astype('float32')
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, embeddings
    
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index, embeddings


def retrieve_top_k_with_metadata(query, chunks, metadata, index, k=7):
    if model is None:
        # Fallback: return first k chunks
        logger.warning("[kg290] Using fallback retrieval due to model loading failure")
        return chunks[:k], metadata[:k], np.array([0.5] * k)
    
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


async def call_groq_api_with_rate_limiting(payload):
    """Enhanced API call with rate limiting and caching"""
    try:
        # Generate cache key
        cache_key = generate_context_hash([
            payload.get("model", ""),
            str(payload.get("messages", [])),
            str(payload.get("temperature", 0.1))
        ])
        
        # Define the API call function
        def api_call():
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload, 
                headers=headers, 
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        
        # Execute with rate limiting
        result = await rate_limiter.execute_with_rate_limit(
            api_call, 
            cache_key
        )
        
        # Update stats
        global request_stats
        request_stats["total_requests"] += 1
        
        if result.cached:
            request_stats["cached_responses"] += 1
            logger.info("[kg290] Used cached response")
        
        if result.status.value == "rate_limited":
            request_stats["rate_limited"] += 1
            logger.warning(f"[kg290] Rate limited after {result.attempts} attempts")
            
            # Return fallback response
            return generate_fallback_response(payload, "Rate limit exceeded")
        
        if result.status.value == "error":
            request_stats["errors"] += 1
            logger.error(f"[kg290] API error: {result.error_message}")
            
            # Return fallback response
            return generate_fallback_response(payload, result.error_message)
        
        return result.data
        
    except Exception as e:
        request_stats["errors"] += 1
        logger.error(f"[kg290] Critical API error: {str(e)}")
        return generate_fallback_response(payload, str(e))


def generate_fallback_response(payload, error_message):
    """Generate fallback response when API fails"""
    logger.info("[kg290] Generating fallback response")
    
    # Extract query from payload
    messages = payload.get("messages", [])
    query_context = ""
    for message in messages:
        if message.get("role") == "user":
            query_context = message.get("content", "")[:200]
            break
    
    # Simple fallback responses based on common patterns
    if "grace period" in query_context.lower():
        return "Grace periods are typically specified in the policy terms. Please refer to your specific policy document for exact details."
    elif "waiting period" in query_context.lower():
        return "Waiting periods vary by coverage type. Please check your policy schedule for specific waiting period information."
    elif "premium" in query_context.lower():
        return "Premium details are outlined in your policy schedule. Contact your insurance provider for specific premium information."
    elif "coverage" in query_context.lower():
        return "Coverage details are specified in your policy terms and conditions. Review your policy document for comprehensive coverage information."
    else:
        return "Please refer to your policy document for specific details, or contact your insurance provider for assistance."


def call_groq_api(payload):
    """Legacy sync function - deprecated, use async version"""
    logger.warning("[kg290] Using deprecated sync API call - consider upgrading to async")
    
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
        logger.error(f"[kg290] Legacy API call failed: {str(e)}")
        return generate_fallback_response(payload, str(e))


async def scout_analysis(query, context_chunks, distances):
    """Fast analysis with Llama 4 Scout - Optimized for concise responses"""

    # Use optimized context for better token efficiency  
    context = "\n\n".join(context_chunks[:4])  # Reduced from 5 to 4

    payload = {
        "model": SCOUT_MODEL,
        "messages": [
            {"role": "system", "content": """You are an expert insurance policy analyst optimized for concise, accurate responses.

CRITICAL INSTRUCTIONS:
- Provide DIRECT, CONCISE answers without verbose explanations
- Focus on specific facts, numbers, and policy clauses
- Use the format: "A grace period of X days..." or "The waiting period is X months..."
- Avoid unnecessary context or background information
- Extract exact policy language when possible

Respond in precise JSON format for structured processing."""},
            {"role": "user", "content": f"""
Query: {query}

Policy Context:
{context}

Confidence Scores: {distances.tolist()[:4]}

Provide a CONCISE analysis in EXACT JSON format:
{{
    "needs_deep_reasoning": boolean,
    "confidence_score": float between 0.0-1.0,
    "preliminary_answer": "DIRECT, concise answer focusing on specific facts",
    "reasoning": "brief step-by-step analysis",
    "complexity_level": "simple|medium|complex",
    "relevant_clauses": ["specific clause references"],
    "key_facts": ["extracted key facts with numbers/periods"]
}}

OPTIMIZATION RULES:
- Set needs_deep_reasoning: false if answer is straightforward with confidence > 0.8
- preliminary_answer should be max 2 sentences for simple queries
- Focus on extracting specific periods, amounts, conditions
- Include exact policy language for facts
- Prioritize accuracy over completeness for time-sensitive evaluation
"""}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.05,  # Reduced for more consistent responses
        "max_tokens": 1500  # Reduced from 2000
    }

    return await call_groq_api_with_rate_limiting(payload)


async def deep_reasoning_analysis(query, context_chunks, metadata):
    """Comprehensive analysis with Llama 3.1 70B - Optimized for concise responses"""

    context = "\n\n".join(context_chunks)

    logger.info(f"[kg290] Starting optimized deep reasoning analysis...")
    logger.info(f"[kg290] Context length: {len(context)} characters, {len(context_chunks)} chunks")

    payload = {
        "model": DEEP_MODEL,
        "messages": [
            {"role": "system", "content": """You are a senior insurance policy expert providing concise, accurate analysis.

CRITICAL OPTIMIZATION REQUIREMENTS:
- Provide DIRECT answers without verbose introductions
- Use specific policy language and exact numbers/periods  
- Format responses as: "A grace period of thirty days is provided..." 
- Focus on actionable facts and specific conditions
- Avoid generic explanations or background context
- Extract and highlight key policy clauses

Optimize for evaluation criteria: accuracy, token efficiency, explainability."""},
            {"role": "user", "content": f"""
Query: {query}

Complete Policy Context:
{context}

Metadata: {json.dumps(metadata, indent=2)}

Provide CONCISE analysis in EXACT JSON format:
{{
    "detailed_answer": "Direct, specific answer with exact policy language",
    "supporting_clauses": ["exact clause text with numbers"],
    "conditions_and_limitations": ["specific conditions only"],
    "confidence_score": float between 0.0-1.0,
    "reasoning_steps": ["brief factual steps"],
    "key_facts": ["specific facts with numbers/periods/amounts"],
    "analysis_depth": "concise|focused|comprehensive",
    "certainty_level": "high|medium|low"
}}

OPTIMIZATION TARGETS:
- detailed_answer: Maximum 3 sentences for simple queries
- Focus on specific policy provisions with exact language
- Include numbers, time periods, and monetary amounts
- Reference specific clauses by number when available
- Prioritize accuracy and specificity over completeness
- Set confidence_score based on factual clarity (aim for 0.85+ when facts are clear)
"""}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,  # Reduced for consistency
        "max_tokens": 2500  # Reduced from 3000
    }

    logger.info(f"[kg290] Calling optimized Llama 3.1 70B API...")

    try:
        response = await call_groq_api_with_rate_limiting(payload)
        logger.info(f"[kg290] Deep model response received successfully")
        
        # Try to parse and validate the response
        try:
            parsed_response = json.loads(response)
            confidence = parsed_response.get("confidence_score", "unknown")
            analysis_depth = parsed_response.get("analysis_depth", "unknown")
            logger.info(f"[kg290] Deep analysis confidence: {confidence}")
            logger.info(f"[kg290] Analysis depth: {analysis_depth}")

        except json.JSONDecodeError:
            logger.warning(f"[kg290] Deep model response is not valid JSON, but proceeding...")

        return response

    except Exception as e:
        logger.error(f"[kg290] Error in deep reasoning analysis: {str(e)}")

        # Return optimized fallback response
        fallback_response = {
            "detailed_answer": generate_fallback_response(payload, str(e)),
            "supporting_clauses": [],
            "conditions_and_limitations": ["Analysis incomplete due to technical error"],
            "confidence_score": 0.4,
            "reasoning_steps": ["Deep analysis failed", "Using fallback response"],
            "key_facts": [],
            "analysis_depth": "failed",
            "certainty_level": "low",
            "error": str(e)
        }

        return json.dumps(fallback_response)


async def process_single_query(query: str, chunks, metadata, index):
    """Optimized dual LLM processing with enhanced response optimization"""
    start_time = time.time()
    
    try:
        logger.info(f"[kg290] Processing optimized query: {query[:100]}...")

        # Retrieve relevant chunks
        top_chunks, chunk_metadata, distances = retrieve_top_k_with_metadata(
            query, chunks, metadata, index, k=6  # Reduced from 7 for efficiency
        )

        # Scout analysis with Llama 4 Scout
        scout_response = await scout_analysis(query, top_chunks, distances)

        try:
            scout_data = json.loads(scout_response)
            scout_confidence = scout_data.get('confidence_score', 0.0)
            logger.info(f"[kg290] Scout analysis complete. Confidence: {scout_confidence}")

            # Enhanced decision logic for optimization
            needs_deep = scout_data.get("needs_deep_reasoning", False)
            confidence = scout_data.get("confidence_score", 0.0)
            preliminary_answer = scout_data.get("preliminary_answer", "")

            # Optimize preliminary answer if available
            if preliminary_answer and not needs_deep and confidence > 0.8:
                optimized_answer = optimize_for_hackathon(preliminary_answer, query)
                
                processing_time = time.time() - start_time
                request_stats["optimization_applied"] += 1
                
                logger.info(f"[kg290] Using optimized Scout response (confidence: {confidence})")
                return {
                    "answer": optimized_answer,
                    "reasoning": scout_data.get("reasoning", ""),
                    "confidence": confidence,
                    "model_used": "scout_optimized",
                    "complexity": scout_data.get("complexity_level", "simple"),
                    "optimization_applied": True,
                    "processing_time": processing_time,
                    "cached": False
                }
            else:
                logger.info(f"[kg290] Escalating to deep reasoning (scout confidence: {confidence})")

                # Deep analysis with optimization
                deep_response = await deep_reasoning_analysis(query, top_chunks, chunk_metadata)

                try:
                    deep_data = json.loads(deep_response)
                    deep_confidence = deep_data.get("confidence_score", 0.8)
                    detailed_answer = deep_data.get("detailed_answer", "Unable to provide detailed analysis")

                    # Apply response optimization
                    optimized_answer = optimize_for_hackathon(detailed_answer, query)
                    processing_time = time.time() - start_time
                    request_stats["optimization_applied"] += 1

                    logger.info(f"[kg290] Deep analysis complete. Confidence: {deep_confidence}")
                    logger.info(f"[kg290] Model progression: Scout({confidence:.3f}) → Deep({deep_confidence:.3f})")

                    return {
                        "answer": optimized_answer,
                        "reasoning": deep_data.get("reasoning_steps", []),
                        "confidence": deep_confidence,
                        "model_used": "dual_llm_optimized",
                        "supporting_clauses": deep_data.get("supporting_clauses", []),
                        "conditions": deep_data.get("conditions_and_limitations", []),
                        "key_facts": deep_data.get("key_facts", []),
                        "scout_confidence": confidence,
                        "deep_confidence": deep_confidence,
                        "analysis_depth": deep_data.get("analysis_depth", "comprehensive"),
                        "optimization_applied": True,
                        "processing_time": processing_time,
                        "cached": False
                    }
                except json.JSONDecodeError as e:
                    logger.warning(f"[kg290] Deep model JSON parsing failed: {str(e)}")
                    
                    # Optimize raw response as fallback
                    optimized_answer = optimize_for_hackathon(deep_response.strip(), query)
                    processing_time = time.time() - start_time
                    
                    return {
                        "answer": optimized_answer,
                        "reasoning": "Processed with deep reasoning model (JSON parsing failed)",
                        "confidence": 0.75,
                        "model_used": "deep_fallback_optimized",
                        "scout_confidence": confidence,
                        "parsing_error": str(e),
                        "optimization_applied": True,
                        "processing_time": processing_time,
                        "cached": False
                    }

        except json.JSONDecodeError as e:
            logger.warning(f"[kg290] Scout JSON parsing failed: {str(e)}")
            
            # Optimize raw scout response as fallback
            optimized_answer = optimize_for_hackathon(scout_response.strip(), query)
            processing_time = time.time() - start_time
            
            return {
                "answer": optimized_answer,
                "reasoning": "Processed with Llama 4 Scout (JSON parsing failed)",
                "confidence": 0.65,
                "model_used": "scout_fallback_optimized",
                "parsing_error": str(e),
                "optimization_applied": True,
                "processing_time": processing_time,
                "cached": False
            }

    except Exception as e:
        logger.error(f"[kg290] Critical error in query processing: {str(e)}")
        processing_time = time.time() - start_time
        
        # Generate optimized fallback response
        fallback_content = generate_fallback_response({"messages": [{"role": "user", "content": query}]}, str(e))
        optimized_fallback = optimize_for_hackathon(fallback_content, query)
        
        return {
            "answer": optimized_fallback,
            "reasoning": "System error occurred during processing",
            "confidence": 0.3,
            "model_used": "error_fallback_optimized",
            "error_details": str(e),
            "optimization_applied": True,
            "processing_time": processing_time,
            "cached": False
        }
@app.get("/")
def root():
    return {
        "message": "Insurance Policy Assistant API v3.0 - Optimized",
        "status": "running",
        "user": "kg290",
        "models": {
            "scout": SCOUT_MODEL,
            "deep": DEEP_MODEL
        },
        "optimizations": {
            "rate_limiting": True,
            "response_caching": True,
            "response_optimization": True,
            "fallback_responses": True
        },
        "stats": request_stats
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "user": "kg290",
        "scout_model": SCOUT_MODEL,
        "deep_model": DEEP_MODEL,
        "rate_limiter_stats": rate_limiter.get_stats(),
        "optimizations_active": True
    }


@app.get("/stats")
def get_stats():
    """Get system statistics"""
    return {
        "request_stats": request_stats,
        "rate_limiter_stats": rate_limiter.get_stats(),
        "cache_stats": rate_limiter.cache.get_stats() if rate_limiter.cache else None,
        "user": "kg290",
        "timestamp": time.time()
    }


@app.post("/ask")
async def ask_query(req: QueryRequest):
    """Development endpoint for single queries with optimized processing"""
    try:
        start_time = time.time()
        
        # Use local PDF for development
        pdf_path = "data/Dataset1.pdf"

        if not os.path.exists(pdf_path):
            return {
                "error": f"PDF file not found at {pdf_path}. Please check the file path.",
                "fallback_answer": generate_fallback_response(
                    {"messages": [{"role": "user", "content": req.query}]}, 
                    "PDF not found"
                )
            }

        with open(pdf_path, 'rb') as file:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            raw_text = "\n\n".join([page.get_text() for page in doc])
            doc.close()

        chunks, metadata = chunk_text_with_metadata(raw_text)
        index, _ = create_faiss_index(chunks)

        result = await process_single_query(req.query, chunks, metadata, index)
        processing_time = time.time() - start_time

        return {
            "response": result.get("answer", "No answer generated"),
            "confidence": result.get("confidence", 0.0),
            "model_used": result.get("model_used", "unknown"),
            "reasoning": result.get("reasoning", ""),
            "optimization_applied": result.get("optimization_applied", False),
            "processing_time": processing_time,
            "cached": result.get("cached", False),
            "status": "success",
            "scout_model": SCOUT_MODEL
        }

    except Exception as e:
        logger.error(f"[kg290] Ask endpoint error: {str(e)}")
        return {
            "error": str(e), 
            "status": "failed",
            "fallback_answer": generate_fallback_response(
                {"messages": [{"role": "user", "content": req.query}]}, 
                str(e)
            )
        }


@app.post("/hackrx/run")
async def hackathon_endpoint(request: HackathonRequest):
    """Main hackathon endpoint with full optimization suite"""
    start_time = time.time()
    
    try:
        logger.info(f"[kg290] Processing optimized document from: {request.documents}")
        logger.info(f"[kg290] Using models - Scout: {SCOUT_MODEL}, Deep: {DEEP_MODEL}")

        # Download and process document
        raw_text = extract_pdf_from_url(request.documents)
        chunks, metadata = chunk_text_with_metadata(raw_text)
        index, _ = create_faiss_index(chunks)

        logger.info(f"[kg290] Created {len(chunks)} chunks, processing {len(request.questions)} questions")

        # Process all questions with optimized dual LLM system
        answers = []
        model_usage_stats = {"scout_optimized": 0, "dual_llm_optimized": 0, "fallback_optimized": 0}
        processing_times = []
        cached_count = 0

        for i, question in enumerate(request.questions):
            logger.info(f"[kg290] Processing question {i + 1}/{len(request.questions)} with optimization")
            
            result = await process_single_query(question, chunks, metadata, index)

            # Extract optimized answer
            answer = result.get("answer", "Unable to determine from the policy document.")
            answers.append(answer)

            # Track model usage
            model_used = result.get("model_used", "unknown")
            if "scout_optimized" in model_used:
                model_usage_stats["scout_optimized"] += 1
            elif "dual_llm_optimized" in model_used:
                model_usage_stats["dual_llm_optimized"] += 1
            else:
                model_usage_stats["fallback_optimized"] += 1
            
            # Track performance metrics
            if result.get("processing_time"):
                processing_times.append(result["processing_time"])
            
            if result.get("cached"):
                cached_count += 1

        total_processing_time = time.time() - start_time
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        logger.info(f"[kg290] Completed optimized processing. Model usage: {model_usage_stats}")
        logger.info(f"[kg290] Performance: {cached_count} cached, avg time: {avg_processing_time:.2f}s")

        return {
            "answers": answers,
            "_metadata": {
                "total_questions": len(request.questions),
                "model_usage": model_usage_stats,
                "performance_metrics": {
                    "total_processing_time": total_processing_time,
                    "average_processing_time": avg_processing_time,
                    "cached_responses": cached_count,
                    "optimization_rate": request_stats["optimization_applied"] / request_stats["total_requests"] if request_stats["total_requests"] > 0 else 0
                },
                "scout_model": SCOUT_MODEL,
                "deep_model": DEEP_MODEL,
                "optimizations_enabled": {
                    "rate_limiting": True,
                    "response_caching": True,
                    "response_optimization": True,
                    "fallback_handling": True
                },
                "user": "kg290",
                "version": "3.0.0-optimized"
            }
        }

    except Exception as e:
        logger.error(f"[kg290] Hackathon endpoint error: {str(e)}")
        
        # Generate fallback responses for all questions
        fallback_answers = []
        for question in request.questions:
            fallback_content = generate_fallback_response(
                {"messages": [{"role": "user", "content": question}]}, 
                str(e)
            )
            fallback_answers.append(optimize_for_hackathon(fallback_content, question))
        
        return {
            "answers": fallback_answers,
            "_metadata": {
                "error": str(e),
                "fallback_used": True,
                "total_questions": len(request.questions),
                "user": "kg290",
                "version": "3.0.0-optimized"
            }
        }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"[kg290] Starting optimized server with Scout Model: {SCOUT_MODEL}")
    logger.info(f"[kg290] Optimizations enabled: Rate limiting, Caching, Response optimization")
    logger.info(f"[kg290] Rate limits: {rate_limit_config.requests_per_minute} RPM, {rate_limit_config.requests_per_hour} RPH")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)