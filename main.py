from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import asyncio
import os
import logging
from dotenv import load_dotenv

# Import your robust modules
from embedder import create_embedder
from retriever import create_retriever, RetrievalConfig
from llm_router import create_llm_router
from pdf_parser import create_parser

# Setup logging for hackathon debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Validate environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is not set in .env file")
    
logger.info(f"[kg290] FastAPI Insurance Assistant initializing at {os.environ.get('CURRENT_TIME', '2025-07-30 15:58:43')}")

# Initialize FastAPI
app = FastAPI(
    title="Enhanced Insurance Policy Assistant", 
    version="3.0.0",
    description="Hackathon-optimized LLM-driven policy QA system"
)

# CORS for hackathon demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components - initialized once for efficiency
logger.info("[kg290] Initializing core components...")
embedder = create_embedder("intfloat/e5-base-v2")
llm_router = create_llm_router(api_key)
response_parser = create_parser()

# Model configurations
SCOUT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEEP_MODEL = "llama3-70b-8192"

logger.info(f"[kg290] Components initialized successfully")

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]

class HealthResponse(BaseModel):
    status: str
    models: Dict[str, str]
    timestamp: str
    user: str

def extract_pdf_from_url(pdf_url: str) -> str:
    """Enhanced PDF extraction with better error handling"""
    import requests
    import tempfile
    import fitz
    
    try:
        logger.info(f"[kg290] Downloading PDF from: {pdf_url[:50]}...")
        
        response = requests.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            temp_path = temp_file.name
        
        logger.info(f"[kg290] Processing PDF...")
        doc = fitz.open(temp_path)
        text = "\n\n".join([page.get_text() for page in doc])
        doc.close()
        
        os.unlink(temp_path)
        
        if not text.strip():
            raise ValueError("PDF contains no extractable text")
            
        logger.info(f"[kg290] PDF processed: {len(text)} characters extracted")
        return text
        
    except requests.RequestException as e:
        logger.error(f"[kg290] PDF download failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        logger.error(f"[kg290] PDF processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

async def process_policy_query(query: str, chunks: List[str], metadata: List[Dict], 
                             faiss_index, delay_seconds: float = 0.5) -> Dict[str, Any]:
    """Enhanced query processing with your modular components"""
    try:
        logger.info(f"[kg290] Processing query: {query[:100]}...")
        
        # Rate limiting delay
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        
        # Use your advanced retriever
        config = RetrievalConfig(
            top_k=7,
            score_threshold=0.7,
            max_context_length=120000,  # Llama 4 Scout's 128K context
            rerank=True
        )
        
        retriever = create_retriever(embedder.model, faiss_index, chunks, metadata, config)
        retrieval_result = retriever.retrieve_with_scores(query, config)
        
        logger.info(f"[kg290] Retrieved {len(retrieval_result.chunks)} relevant chunks")
        
        # Context for LLM
        context_preview = "\n\n".join(retrieval_result.chunks[:3])[:2000]
        
        # Use your intelligent routing
        routing_decision = llm_router.analyze_query_complexity(query, context_preview)
        logger.info(f"[kg290] Routing decision: {routing_decision.model_type.value}, confidence: {routing_decision.confidence}")
        
        # Process based on routing
        if routing_decision.model_type.value == "scout":
            llm_response = llm_router.scout_analysis(query, "\n\n".join(retrieval_result.chunks), routing_decision)
        else:
            llm_response = llm_router.deep_analysis(query, "\n\n".join(retrieval_result.chunks))
        
        # Parse response with your robust parser
        if "scout" in llm_response.model_used.lower():
            parsed = response_parser.parse_scout_response(llm_response.content, llm_response.model_used)
        else:
            parsed = response_parser.parse_deep_response(llm_response.content, llm_response.model_used)
        
        return {
            "answer": parsed.answer,
            "confidence": parsed.confidence,
            "confidence_level": parsed.confidence_level.value,
            "reasoning": parsed.reasoning,
            "relevant_clauses": parsed.clauses,
            "conditions": parsed.conditions,
            "model_used": llm_response.model_used,
            "retrieval_score": max(retrieval_result.scores) if retrieval_result.scores else 0.0,
            "chunks_used": len(retrieval_result.chunks),
            "parsing_success": parsed.parsing_success,
            "user": "kg290",
            "timestamp": "2025-07-30 15:58:43"
        }
        
    except Exception as e:
        logger.error(f"[kg290] Query processing failed: {str(e)}")
        return {
            "answer": f"Error processing query: {str(e)}",
            "confidence": 0.0,
            "confidence_level": "low",
            "model_used": "error_handler",
            "parsing_success": False,
            "user": "kg290",
            "timestamp": "2025-07-30 15:58:43"
        }

# FastAPI Routes
@app.get("/", response_model=Dict[str, Any])
def root():
    return {
        "message": "Enhanced Insurance Policy Assistant API v3.0",
        "status": "running",
        "features": [
            "Smart chunking with sentence boundaries",
            "Hybrid semantic + keyword retrieval", 
            "Intelligent LLM routing (Scout/Deep)",
            "Robust response parsing",
            "128K context optimization"
        ],
        "models": {
            "scout": SCOUT_MODEL,
            "deep": DEEP_MODEL,
            "embedder": "intfloat/e5-base-v2"
        },
        "user": "kg290",
        "timestamp": "2025-07-30 15:58:43"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        models={
            "scout": SCOUT_MODEL,
            "deep": DEEP_MODEL,
            "embedder": "intfloat/e5-base-v2"
        },
        timestamp="2025-07-30 15:58:43",
        user="kg290"
    )

@app.post("/ask")
async def ask_query(req: QueryRequest):
    """Development endpoint with local PDF support"""
    try:
        pdf_path = "data/Dataset1.pdf"
        
        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=404, 
                detail=f"PDF file not found at {pdf_path}. Please ensure the file exists."
            )
        
        # Use your embedder for smart processing
        with open(pdf_path, 'rb') as file:
            import fitz
            doc = fitz.open(stream=file.read(), filetype="pdf")
            raw_text = "\n\n".join([page.get_text() for page in doc])
            doc.close()
        
        # Use advanced chunking from embedder
        chunks, metadata = embedder.chunk_text_with_metadata(raw_text, chunk_size=1200, overlap=100)
        index, embeddings = embedder.create_faiss_index(chunks)
        
        result = await process_policy_query(req.query, chunks, metadata, index)
        
        return {
            "response": result.get("answer", "No answer generated"),
            "confidence": result.get("confidence", 0.0),
            "confidence_level": result.get("confidence_level", "low"),
            "model_used": result.get("model_used", "unknown"),
            "reasoning": result.get("reasoning", []),
            "relevant_clauses": result.get("relevant_clauses", []),
            "retrieval_stats": {
                "chunks_processed": len(chunks),
                "chunks_used": result.get("chunks_used", 0),
                "retrieval_score": result.get("retrieval_score", 0.0)
            },
            "status": "success",
            "user": "kg290",
            "timestamp": "2025-07-30 15:58:43"
        }
        
    except Exception as e:
        logger.error(f"[kg290] Ask query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hackrx/run")
async def hackathon_endpoint(request: HackathonRequest):
    """Optimized hackathon endpoint using modular components"""
    try:
        logger.info(f"[kg290] Hackathon request: {len(request.questions)} questions")
        logger.info(f"[kg290] Document URL: {request.documents[:50]}...")
        
        # Extract PDF with robust error handling
        raw_text = extract_pdf_from_url(request.documents)
        
        # Use advanced embedder for optimal chunking
        chunks, metadata = embedder.chunk_text_with_metadata(
            raw_text, 
            chunk_size=1200,  # Optimized for Llama 4 Scout
            overlap=100
        )
        
        # Create optimized FAISS index
        index, embeddings = embedder.create_faiss_index(chunks)
        
        logger.info(f"[kg290] Created {len(chunks)} chunks, processing {len(request.questions)} questions")
        
        # Process all questions with intelligent routing
        answers = []
        processing_stats = {
            "scout_used": 0,
            "deep_used": 0,
            "avg_confidence": 0.0,
            "total_chunks_retrieved": 0
        }
        
        total_confidence = 0.0
        
        for i, question in enumerate(request.questions):
            logger.info(f"[kg290] Processing question {i+1}/{len(request.questions)}")
            
            # Progressive delay for rate limiting
            delay = 0.2 + (i * 0.05)  # Optimized for hackathon speed
            result = await process_policy_query(question, chunks, metadata, index, delay)
            
            answer = result.get("answer", "Unable to determine from the policy document.")
            answers.append(answer)
            
            # Track statistics for evaluation
            confidence = result.get("confidence", 0.0)
            total_confidence += confidence
            processing_stats["total_chunks_retrieved"] += result.get("chunks_used", 0)
            
            if "scout" in result.get("model_used", "").lower():
                processing_stats["scout_used"] += 1
            else:
                processing_stats["deep_used"] += 1
        
        processing_stats["avg_confidence"] = total_confidence / len(request.questions) if request.questions else 0.0
        
        logger.info(f"[kg290] Processing completed. Stats: {processing_stats}")
        
        # Return hackathon-expected format
        return {
            "answers": answers
        }
        
    except Exception as e:
        logger.error(f"[kg290] Hackathon endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_system_stats():
    """Get system statistics for debugging/monitoring"""
    return {
        "embedder_stats": embedder.get_embedding_stats(embedder.model.encode(["test"])),
        "parser_stats": response_parser.get_parsing_stats(),
        "models": {
            "scout": SCOUT_MODEL,
            "deep": DEEP_MODEL
        },
        "user": "kg290",
        "timestamp": "2025-07-30 15:58:43"
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"[kg290] Starting Enhanced Insurance Assistant API")
    logger.info(f"[kg290] Scout Model: {SCOUT_MODEL}")
    logger.info(f"[kg290] Deep Model: {DEEP_MODEL}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
