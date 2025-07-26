"""
Enhanced LLM Routing module with Llama 4 Scout integration
Author: kg290
Date: 2025-07-26 11:40:07
"""

import json
import requests
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    SCOUT = "scout"
    DEEP = "deep"
    FALLBACK = "fallback"


class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class RoutingDecision:
    model_type: ModelType
    confidence: float
    reasoning: str
    complexity: ComplexityLevel
    estimated_tokens: int
    needs_deep_reasoning: bool


@dataclass
class LLMResponse:
    content: str
    model_used: str
    confidence: float
    reasoning: str
    token_usage: Dict[str, int]
    metadata: Dict[str, Any]


class LLMRouter:
    def __init__(self, api_key: str):
        """
        Initialize LLM Router with Llama 4 Scout as primary scout model
        """
        self.api_key = api_key

        # Updated model configurations for Llama 4 Scout
        self.models = {
            "scout": {
                "name": "meta-llama/llama-4-scout-17b-16e-instruct",  # Updated from llama3-8b-8192
                "max_tokens": 128000,  # 128K context window
                "cost_per_token": 0.0001,  # Estimated cost
                "capabilities": ["json_mode", "structured_output", "fast_inference", "moe_architecture"]
            },
            "deep": {
                "name": "llama3-70b-8192",
                "max_tokens": 8192,
                "cost_per_token": 0.0008,
                "capabilities": ["complex_reasoning", "detailed_analysis", "legal_interpretation"]
            }
        }

        logger.info(f"[kg290] LLMRouter initialized with Scout: {self.models['scout']['name']}")

    def call_groq_api(self, payload: Dict) -> str:
        """
        Enhanced API call with better error handling
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            logger.info(f"[kg290] Calling Groq API with model: {payload.get('model', 'unknown')}")

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            logger.info(f"[kg290] API call successful, response length: {len(content)}")
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"[kg290] API request failed: {str(e)}")
            raise Exception(f"LLM API call failed: {str(e)}")
        except (KeyError, IndexError) as e:
            logger.error(f"[kg290] API response parsing failed: {str(e)}")
            raise Exception(f"Invalid API response format: {str(e)}")

    def analyze_query_complexity(self, query: str, context_preview: str) -> RoutingDecision:
        """
        Enhanced query analysis using Llama 4 Scout for intelligent routing
        """
        logger.info(f"[kg290] Analyzing query complexity with Llama 4 Scout")

        payload = {
            "model": self.models["scout"]["name"],  # meta-llama/llama-4-scout-17b-16e-instruct
            "messages": [
                {
                    "role": "system",
                    "content": """You are an expert insurance policy analyst using Llama 4 Scout capabilities.
                    Analyze queries to determine optimal processing strategy.

                    Your job is to route queries efficiently while maintaining accuracy."""
                },
                {
                    "role": "user",
                    "content": f"""
Query: {query}

Context Preview: {context_preview[:2000]}

Analyze this insurance query and determine processing strategy.

Respond with EXACT JSON format:
{{
    "complexity_level": "simple|medium|complex",
    "needs_deep_reasoning": boolean,
    "confidence_score": float (0.0-1.0),
    "reasoning": "detailed explanation",
    "estimated_tokens": integer,
    "key_factors": ["factor1", "factor2"],
    "query_type": "factual|calculation|interpretation|comparison",
    "can_answer_directly": boolean
}}

Classification Rules:
- SIMPLE: Direct factual questions, single clause lookups, basic definitions
- MEDIUM: Questions requiring 2-3 clause comparisons, simple calculations
- COMPLEX: Multi-condition analysis, legal interpretations, complex calculations

Set needs_deep_reasoning=true for COMPLEX queries or when confidence < 0.8
"""
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 1000
        }

        try:
            response = self.call_groq_api(payload)
            analysis = json.loads(response)

            # Create routing decision
            complexity = ComplexityLevel(analysis.get("complexity_level", "medium"))
            needs_deep = analysis.get("needs_deep_reasoning", False)
            confidence = analysis.get("confidence_score", 0.5)

            # Determine model type
            if needs_deep or complexity == ComplexityLevel.COMPLEX or confidence < 0.8:
                model_type = ModelType.DEEP
            else:
                model_type = ModelType.SCOUT

            decision = RoutingDecision(
                model_type=model_type,
                confidence=confidence,
                reasoning=analysis.get("reasoning", "No reasoning provided"),
                complexity=complexity,
                estimated_tokens=analysis.get("estimated_tokens", 500),
                needs_deep_reasoning=needs_deep
            )

            logger.info(f"[kg290] Routing decision: {model_type.value}, confidence: {confidence}")
            return decision

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[kg290] Complexity analysis failed, using fallback routing: {str(e)}")

            # Fallback routing logic
            return RoutingDecision(
                model_type=ModelType.DEEP,
                confidence=0.5,
                reasoning="Fallback routing due to analysis failure",
                complexity=ComplexityLevel.MEDIUM,
                estimated_tokens=800,
                needs_deep_reasoning=True
            )

    def scout_analysis(self, query: str, context: str, routing_decision: RoutingDecision) -> LLMResponse:
        """
        Enhanced scout analysis with Llama 4 Scout
        """
        logger.info(f"[kg290] Performing scout analysis with Llama 4 Scout")

        payload = {
            "model": self.models["scout"]["name"],
            "messages": [
                {
                    "role": "system",
                    "content": """You are an expert insurance policy analyst powered by Llama 4 Scout.
                    Provide accurate, concise answers with high confidence when possible.

                    Use your advanced MoE architecture for efficient processing."""
                },
                {
                    "role": "user",
                    "content": f"""
Query: {query}

Policy Context:
{context}

Complexity Assessment: {routing_decision.complexity.value}
Confidence Threshold: 0.85

Provide analysis in EXACT JSON format:
{{
    "answer": "detailed and accurate answer",
    "confidence_score": float (0.0-1.0),
    "reasoning_steps": ["step1", "step2", "step3"],
    "relevant_clauses": ["clause references"],
    "certainty_level": "high|medium|low",
    "requires_escalation": boolean,
    "additional_context_needed": boolean
}}

Instructions:
- Provide complete answers when confidence > 0.85
- Set requires_escalation=true if answer needs deeper analysis
- Reference specific policy clauses when available
- Use structured reasoning approach
"""
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 2000
        }

        try:
            response_content = self.call_groq_api(payload)
            response_data = json.loads(response_content)

            return LLMResponse(
                content=response_data.get("answer", "Unable to provide answer"),
                model_used=self.models["scout"]["name"],
                confidence=response_data.get("confidence_score", 0.0),
                reasoning=str(response_data.get("reasoning_steps", [])),
                token_usage={"estimated": routing_decision.estimated_tokens},
                metadata={
                    "complexity": routing_decision.complexity.value,
                    "requires_escalation": response_data.get("requires_escalation", False),
                    "relevant_clauses": response_data.get("relevant_clauses", []),
                    "user": "kg290",
                    "timestamp": "2025-07-26 11:40:07"
                }
            )

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"[kg290] Scout analysis failed: {str(e)}")
            return LLMResponse(
                content=response_content if 'response_content' in locals() else "Analysis failed",
                model_used=self.models["scout"]["name"],
                confidence=0.0,
                reasoning=f"Scout analysis error: {str(e)}",
                token_usage={"estimated": 0},
                metadata={"error": str(e), "user": "kg290"}
            )

    def deep_analysis(self, query: str, context: str, scout_result: Optional[LLMResponse] = None) -> LLMResponse:
        """
        Deep reasoning analysis with Llama 3.1 70B
        """
        logger.info(f"[kg290] Performing deep analysis with Llama 3.1 70B")

        scout_context = ""
        if scout_result:
            scout_context = f"\nScout Analysis: {scout_result.content}\nScout Confidence: {scout_result.confidence}"

        payload = {
            "model": self.models["deep"]["name"],
            "messages": [
                {
                    "role": "system",
                    "content": """You are a senior insurance policy expert providing comprehensive analysis.
                    Deliver detailed, accurate answers with thorough reasoning and evidence.

                    Focus on precision, completeness, and actionable insights."""
                },
                {
                    "role": "user",
                    "content": f"""
Query: {query}

Complete Policy Context:
{context}
{scout_context}

Provide comprehensive analysis in EXACT JSON format:
{{
    "detailed_answer": "comprehensive and precise answer",
    "confidence_score": float (0.0-1.0),
    "supporting_evidence": ["evidence1", "evidence2"],
    "relevant_clauses": ["exact clause references"],
    "conditions_and_limitations": ["condition1", "condition2"],
    "reasoning_chain": ["step1", "step2", "step3"],
    "risk_factors": ["risk1", "risk2"],
    "recommendations": ["rec1", "rec2"],
    "additional_considerations": "important notes"
}}

Requirements:
- Provide complete, actionable answers
- Reference specific policy sections
- Include all relevant conditions and limitations
- Explain complex concepts clearly
- Maintain high accuracy standards
"""
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": 3000
        }

        try:
            response_content = self.call_groq_api(payload)
            response_data = json.loads(response_content)

            return LLMResponse(
                content=response_data.get("detailed_answer", "Unable to provide detailed analysis"),
                model_used=self.models["deep"]["name"],
                confidence=response_data.get("confidence_score", 0.8),
                reasoning=str(response_data.get("reasoning_chain", [])),
                token_usage={"estimated": 2000},
                metadata={
                    "supporting_evidence": response_data.get("supporting_evidence", []),
                    "relevant_clauses": response_data.get("relevant_clauses", []),
                    "conditions": response_data.get("conditions_and_limitations", []),
                    "recommendations": response_data.get("recommendations", []),
                    "user": "kg290",
                    "timestamp": "2025-07-26 11:40:07"
                }
            )

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"[kg290] Deep analysis failed: {str(e)}")
            return LLMResponse(
                content=response_content if 'response_content' in locals() else "Deep analysis failed",
                model_used=self.models["deep"]["name"],
                confidence=0.0,
                reasoning=f"Deep analysis error: {str(e)}",
                token_usage={"estimated": 0},
                metadata={"error": str(e), "user": "kg290"}
            )


# Factory function
def create_llm_router(api_key: str) -> LLMRouter:
    """Factory function to create LLM router instance"""
    logger.info(f"[kg290] Creating LLMRouter instance")
    return LLMRouter(api_key)