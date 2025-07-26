"""
Enhanced Response Parser module for Llama 4 Scout outputs
Author: kg290
Date: 2025-07-26 11:40:07
"""

import json
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseType(Enum):
    SCOUT_RESPONSE = "scout"
    DEEP_RESPONSE = "deep"
    ERROR_RESPONSE = "error"
    FALLBACK_RESPONSE = "fallback"


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ParsedResponse:
    answer: str
    confidence: float
    confidence_level: ConfidenceLevel
    reasoning: List[str]
    evidence: List[str]
    clauses: List[str]
    conditions: List[str]
    metadata: Dict[str, Any]
    response_type: ResponseType
    parsing_success: bool
    raw_response: str


class ResponseParser:
    def __init__(self):
        """
        Initialize response parser optimized for Llama 4 Scout structured outputs
        """
        logger.info(f"[kg290] ResponseParser initialized for Llama 4 Scout integration")

        # JSON parsing patterns for different response types
        self.scout_schema = {
            "answer": str,
            "confidence_score": float,
            "reasoning_steps": list,
            "relevant_clauses": list,
            "certainty_level": str,
            "requires_escalation": bool
        }

        self.deep_schema = {
            "detailed_answer": str,
            "confidence_score": float,
            "supporting_evidence": list,
            "relevant_clauses": list,
            "conditions_and_limitations": list,
            "reasoning_chain": list,
            "recommendations": list
        }

        # Confidence thresholds
        self.confidence_thresholds = {
            ConfidenceLevel.HIGH: 0.85,
            ConfidenceLevel.MEDIUM: 0.65,
            ConfidenceLevel.LOW: 0.0
        }

    def parse_scout_response(self, raw_response: str,
                             model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> ParsedResponse:
        """
        Parse Llama 4 Scout response with enhanced structured output handling
        """
        logger.info(f"[kg290] Parsing scout response from {model_name}")

        try:
            # Try to parse as JSON first (Llama 4 Scout supports structured outputs)
            response_data = json.loads(raw_response)

            # Extract core fields
            answer = response_data.get("answer", "")
            confidence = float(response_data.get("confidence_score", 0.0))
            reasoning_steps = response_data.get("reasoning_steps", [])
            relevant_clauses = response_data.get("relevant_clauses", [])
            requires_escalation = response_data.get("requires_escalation", False)

            # Determine confidence level
            confidence_level = self._determine_confidence_level(confidence)

            # Enhanced metadata for scout responses
            metadata = {
                "model_used": model_name,
                "response_type": "scout_structured",
                "requires_escalation": requires_escalation,
                "certainty_level": response_data.get("certainty_level", "medium"),
                "additional_context_needed": response_data.get("additional_context_needed", False),
                "parsing_method": "json_structured",
                "user": "kg290",
                "timestamp": "2025-07-26 11:40:07"
            }

            return ParsedResponse(
                answer=answer,
                confidence=confidence,
                confidence_level=confidence_level,
                reasoning=reasoning_steps if isinstance(reasoning_steps, list) else [str(reasoning_steps)],
                evidence=[],  # Scout responses typically don't include detailed evidence
                clauses=relevant_clauses if isinstance(relevant_clauses, list) else [str(relevant_clauses)],
                conditions=[],  # Detailed conditions usually come from deep analysis
                metadata=metadata,
                response_type=ResponseType.SCOUT_RESPONSE,
                parsing_success=True,
                raw_response=raw_response
            )

        except json.JSONDecodeError:
            logger.warning(f"[kg290] Scout response JSON parsing failed, attempting text extraction")
            return self._parse_text_response(raw_response, ResponseType.SCOUT_RESPONSE, model_name)

        except Exception as e:
            logger.error(f"[kg290] Scout response parsing error: {str(e)}")
            return self._create_error_response(raw_response, str(e), model_name)

    def parse_deep_response(self, raw_response: str, model_name: str = "llama3-70b-8192") -> ParsedResponse:
        """
        Parse deep reasoning model response with comprehensive data extraction
        """
        logger.info(f"[kg290] Parsing deep response from {model_name}")

        try:
            response_data = json.loads(raw_response)

            # Extract comprehensive fields
            answer = response_data.get("detailed_answer", "")
            confidence = float(response_data.get("confidence_score", 0.8))
            supporting_evidence = response_data.get("supporting_evidence", [])
            relevant_clauses = response_data.get("relevant_clauses", [])
            conditions = response_data.get("conditions_and_limitations", [])
            reasoning_chain = response_data.get("reasoning_chain", [])
            recommendations = response_data.get("recommendations", [])

            confidence_level = self._determine_confidence_level(confidence)

            # Enhanced metadata for deep responses
            metadata = {
                "model_used": model_name,
                "response_type": "deep_comprehensive",
                "recommendations": recommendations,
                "risk_factors": response_data.get("risk_factors", []),
                "additional_considerations": response_data.get("additional_considerations", ""),
                "parsing_method": "json_structured",
                "user": "kg290",
                "timestamp": "2025-07-26 11:40:07"
            }

            return ParsedResponse(
                answer=answer,
                confidence=confidence,
                confidence_level=confidence_level,
                reasoning=reasoning_chain if isinstance(reasoning_chain, list) else [str(reasoning_chain)],
                evidence=supporting_evidence if isinstance(supporting_evidence, list) else [str(supporting_evidence)],
                clauses=relevant_clauses if isinstance(relevant_clauses, list) else [str(relevant_clauses)],
                conditions=conditions if isinstance(conditions, list) else [str(conditions)],
                metadata=metadata,
                response_type=ResponseType.DEEP_RESPONSE,
                parsing_success=True,
                raw_response=raw_response
            )

        except json.JSONDecodeError:
            logger.warning(f"[kg290] Deep response JSON parsing failed, attempting text extraction")
            return self._parse_text_response(raw_response, ResponseType.DEEP_RESPONSE, model_name)

        except Exception as e:
            logger.error(f"[kg290] Deep response parsing error: {str(e)}")
            return self._create_error_response(raw_response, str(e), model_name)

    def _parse_text_response(self, raw_response: str, response_type: ResponseType, model_name: str) -> ParsedResponse:
        """
        Fallback text parsing for non-JSON responses
        """
        logger.info(f"[kg290] Performing fallback text parsing")

        # Extract answer (assuming first paragraph or sentence)
        answer_match = re.search(r'^(.+?)(?:\n\n|\n(?=[A-Z])|$)', raw_response.strip(), re.MULTILINE | re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else raw_response.strip()

        # Try to extract confidence if mentioned
        confidence_match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)', raw_response, re.IGNORECASE)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.6

        # Extract clauses (look for patterns like "clause 3.2", "section 4.1", etc.)
        clause_patterns = [
            r'clause\s+\d+(?:\.\d+)*',
            r'section\s+\d+(?:\.\d+)*',
            r'paragraph\s+\d+(?:\.\d+)*',
            r'article\s+\d+(?:\.\d+)*'
        ]

        clauses = []
        for pattern in clause_patterns:
            clauses.extend(re.findall(pattern, raw_response, re.IGNORECASE))

        # Extract reasoning (look for numbered points, bullets, etc.)
        reasoning_patterns = [
            r'\d+\.\s+([^\n]+)',
            r'[•\-\*]\s+([^\n]+)',
            r'(?:because|since|as|due to)\s+([^\n.]+)'
        ]

        reasoning = []
        for pattern in reasoning_patterns:
            reasoning.extend(re.findall(pattern, raw_response, re.IGNORECASE))

        confidence_level = self._determine_confidence_level(confidence)

        metadata = {
            "model_used": model_name,
            "response_type": f"{response_type.value}_text_parsed",
            "parsing_method": "regex_extraction",
            "extraction_patterns_used": len([p for p in clause_patterns + reasoning_patterns]),
            "user": "kg290",
            "timestamp": "2025-07-26 11:40:07"
        }

        return ParsedResponse(
            answer=answer,
            confidence=confidence,
            confidence_level=confidence_level,
            reasoning=reasoning[:5],  # Limit to top 5 reasoning points
            evidence=[],
            clauses=list(set(clauses))[:10],  # Deduplicate and limit
            conditions=[],
            metadata=metadata,
            response_type=ResponseType.FALLBACK_RESPONSE,
            parsing_success=True,
            raw_response=raw_response
        )

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """
        Determine confidence level based on numeric confidence score
        """
        if confidence >= self.confidence_thresholds[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif confidence >= self.confidence_thresholds[ConfidenceLevel.MEDIUM]:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _create_error_response(self, raw_response: str, error_msg: str, model_name: str) -> ParsedResponse:
        """
        Create error response for parsing failures
        """
        logger.error(f"[kg290] Creating error response: {error_msg}")

        return ParsedResponse(
            answer="Unable to parse response properly. Please try again.",
            confidence=0.0,
            confidence_level=ConfidenceLevel.LOW,
            reasoning=[f"Parsing error: {error_msg}"],
            evidence=[],
            clauses=[],
            conditions=[],
            metadata={
                "model_used": model_name,
                "response_type": "error",
                "error_message": error_msg,
                "parsing_method": "error_handling",
                "user": "kg290",
                "timestamp": "2025-07-26 11:40:07"
            },
            response_type=ResponseType.ERROR_RESPONSE,
            parsing_success=False,
            raw_response=raw_response
        )

    def format_for_api_response(self, parsed_response: ParsedResponse, include_metadata: bool = False) -> Dict[
        str, Any]:
        """
        Format parsed response for API output
        """
        response = {
            "answer": parsed_response.answer,
            "confidence": parsed_response.confidence,
            "confidence_level": parsed_response.confidence_level.value,
            "model_used": parsed_response.metadata.get("model_used", "unknown"),
            "parsing_success": parsed_response.parsing_success
        }

        # Add optional fields if they exist
        if parsed_response.reasoning:
            response["reasoning"] = parsed_response.reasoning

        if parsed_response.clauses:
            response["relevant_clauses"] = parsed_response.clauses

        if parsed_response.conditions:
            response["conditions_and_limitations"] = parsed_response.conditions

        if parsed_response.evidence:
            response["supporting_evidence"] = parsed_response.evidence

        if include_metadata:
            response["metadata"] = parsed_response.metadata

        return response

    def get_parsing_stats(self) -> Dict[str, Any]:
        """
        Get parser statistics and configuration
        """
        return {
            "supported_models": [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama3-70b-8192"
            ],
            "supported_response_types": [t.value for t in ResponseType],
            "confidence_thresholds": {
                level.value: threshold for level, threshold in self.confidence_thresholds.items()
            },
            "parsing_methods": ["json_structured", "regex_extraction", "error_handling"],
            "user": "kg290",
            "timestamp": "2025-07-26 11:40:07"
        }


# Factory functions
def create_parser() -> ResponseParser:
    """Factory function to create parser instance"""
    logger.info(f"[kg290] Creating ResponseParser instance")
    return ResponseParser()


def parse_llm_response(raw_response: str, model_name: str, response_type: str = "auto") -> ParsedResponse:
    """
    Convenience function to parse LLM response based on model type
    """
    parser = create_parser()

    if "scout" in model_name.lower() or "llama-4" in model_name.lower():
        return parser.parse_scout_response(raw_response, model_name)
    elif "70b" in model_name.lower() or response_type == "deep":
        return parser.parse_deep_response(raw_response, model_name)
    else:
        # Auto-detect based on content
        try:
            data = json.loads(raw_response)
            if "detailed_answer" in data:
                return parser.parse_deep_response(raw_response, model_name)
            else:
                return parser.parse_scout_response(raw_response, model_name)
        except:
            return parser.parse_scout_response(raw_response, model_name)