"""
Response Optimization module for concise, accurate answers
Author: kg290
Date: 2025-07-26
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseStyle(Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    STRUCTURED = "structured"


class FactType(Enum):
    TIME_PERIOD = "time_period"
    MONETARY_AMOUNT = "monetary_amount"
    PERCENTAGE = "percentage"
    CONDITION = "condition"
    REQUIREMENT = "requirement"


@dataclass
class ExtractedFact:
    type: FactType
    value: str
    context: str
    confidence: float
    clause_reference: Optional[str] = None


@dataclass
class OptimizedResponse:
    concise_answer: str
    key_facts: List[ExtractedFact]
    clause_references: List[str]
    confidence: float
    reasoning: List[str]
    fallback_answer: Optional[str] = None
    original_response: Optional[str] = None


class ResponseOptimizer:
    """Optimizes LLM responses for conciseness and accuracy"""
    
    def __init__(self):
        self.fact_extractors = {
            FactType.TIME_PERIOD: self._extract_time_periods,
            FactType.MONETARY_AMOUNT: self._extract_monetary_amounts,
            FactType.PERCENTAGE: self._extract_percentages,
            FactType.CONDITION: self._extract_conditions,
            FactType.REQUIREMENT: self._extract_requirements
        }
        
        # Patterns for fact extraction
        self.patterns = {
            'time_periods': [
                r'(\d+)\s+(days?|months?|years?)',
                r'(thirty|sixty|ninety|one hundred twenty)\s+(days?)',
                r'(\d+)\s+(day|month|year)\s+period',
                r'grace\s+period\s+of\s+(\d+|\w+)\s+(days?|months?)',
                r'waiting\s+period\s+of\s+(\d+|\w+)\s+(days?|months?|years?)'
            ],
            'monetary': [
                r'\$[\d,]+(?:\.\d{2})?',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s+(?:dollars?|USD|INR|rupees?)',
                r'amount\s+of\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'premium\s+of\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
            ],
            'percentages': [
                r'(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s+percent',
                r'rate\s+of\s+(\d+(?:\.\d+)?)\s*%'
            ],
            'conditions': [
                r'provided\s+that\s+([^.]+)',
                r'subject\s+to\s+([^.]+)',
                r'if\s+and\s+only\s+if\s+([^.]+)',
                r'conditional\s+upon\s+([^.]+)'
            ],
            'requirements': [
                r'must\s+([^.]+)',
                r'required\s+to\s+([^.]+)',
                r'shall\s+([^.]+)',
                r'mandatory\s+([^.]+)'
            ]
        }
        
        # Conciseness templates
        self.concise_templates = {
            'grace_period': "Grace period: {period}",
            'waiting_period': "Waiting period: {period}",
            'coverage': "Coverage: {details}",
            'exclusion': "Excluded: {details}",
            'requirement': "Required: {details}",
            'benefit': "Benefit: {details}",
            'premium': "Premium: {amount}",
            'deductible': "Deductible: {amount}"
        }
        
        logger.info("[kg290] ResponseOptimizer initialized")
    
    def optimize_response(self, raw_response: str, 
                         query_type: Optional[str] = None,
                         style: ResponseStyle = ResponseStyle.CONCISE) -> OptimizedResponse:
        """Main optimization function"""
        logger.info(f"[kg290] Optimizing response with style: {style.value}")
        
        try:
            # Try to parse as JSON first
            if raw_response.strip().startswith('{'):
                response_data = json.loads(raw_response)
                content = (response_data.get('answer') or 
                          response_data.get('detailed_answer') or 
                          raw_response)
            else:
                content = raw_response
                response_data = {}
            
            # Extract key facts
            key_facts = self._extract_all_facts(content)
            
            # Extract clause references
            clause_refs = self._extract_clause_references(content)
            
            # Generate concise answer
            if style == ResponseStyle.CONCISE:
                concise_answer = self._generate_concise_answer(content, key_facts, query_type)
            else:
                concise_answer = self._clean_verbose_response(content)
            
            # Extract reasoning
            reasoning = self._extract_reasoning(response_data, content)
            
            # Calculate confidence
            confidence = self._calculate_optimization_confidence(key_facts, clause_refs, content)
            
            # Generate fallback if needed
            fallback = self._generate_fallback_answer(content, key_facts) if confidence < 0.7 else None
            
            return OptimizedResponse(
                concise_answer=concise_answer,
                key_facts=key_facts,
                clause_references=clause_refs,
                confidence=confidence,
                reasoning=reasoning,
                fallback_answer=fallback,
                original_response=raw_response
            )
            
        except Exception as e:
            logger.error(f"[kg290] Response optimization failed: {str(e)}")
            return self._create_fallback_optimization(raw_response, str(e))
    
    def _extract_all_facts(self, content: str) -> List[ExtractedFact]:
        """Extract all types of facts from content"""
        all_facts = []
        
        for fact_type, extractor in self.fact_extractors.items():
            try:
                facts = extractor(content)
                all_facts.extend(facts)
            except Exception as e:
                logger.warning(f"[kg290] Fact extraction failed for {fact_type}: {e}")
        
        # Sort by confidence
        all_facts.sort(key=lambda f: f.confidence, reverse=True)
        return all_facts[:10]  # Limit to top 10 facts
    
    def _extract_time_periods(self, content: str) -> List[ExtractedFact]:
        """Extract time periods (days, months, years)"""
        facts = []
        
        for pattern in self.patterns['time_periods']:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                full_match = match.group(0)
                context = self._get_sentence_context(content, match.start())
                
                # Determine confidence based on context keywords
                confidence = 0.8
                if any(keyword in context.lower() for keyword in ['grace', 'waiting', 'period']):
                    confidence = 0.95
                
                facts.append(ExtractedFact(
                    type=FactType.TIME_PERIOD,
                    value=full_match,
                    context=context,
                    confidence=confidence
                ))
        
        return facts
    
    def _extract_monetary_amounts(self, content: str) -> List[ExtractedFact]:
        """Extract monetary amounts"""
        facts = []
        
        for pattern in self.patterns['monetary']:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                full_match = match.group(0)
                context = self._get_sentence_context(content, match.start())
                
                confidence = 0.85
                if any(keyword in context.lower() for keyword in ['premium', 'deductible', 'amount']):
                    confidence = 0.9
                
                facts.append(ExtractedFact(
                    type=FactType.MONETARY_AMOUNT,
                    value=full_match,
                    context=context,
                    confidence=confidence
                ))
        
        return facts
    
    def _extract_percentages(self, content: str) -> List[ExtractedFact]:
        """Extract percentage values"""
        facts = []
        
        for pattern in self.patterns['percentages']:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                full_match = match.group(0)
                context = self._get_sentence_context(content, match.start())
                
                facts.append(ExtractedFact(
                    type=FactType.PERCENTAGE,
                    value=full_match,
                    context=context,
                    confidence=0.9
                ))
        
        return facts
    
    def _extract_conditions(self, content: str) -> List[ExtractedFact]:
        """Extract conditions and requirements"""
        facts = []
        
        for pattern in self.patterns['conditions']:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                condition = match.group(1).strip()
                context = self._get_sentence_context(content, match.start())
                
                facts.append(ExtractedFact(
                    type=FactType.CONDITION,
                    value=condition,
                    context=context,
                    confidence=0.8
                ))
        
        return facts
    
    def _extract_requirements(self, content: str) -> List[ExtractedFact]:
        """Extract requirements and mandatory items"""
        facts = []
        
        for pattern in self.patterns['requirements']:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                requirement = match.group(1).strip()
                context = self._get_sentence_context(content, match.start())
                
                facts.append(ExtractedFact(
                    type=FactType.REQUIREMENT,
                    value=requirement,
                    context=context,
                    confidence=0.85
                ))
        
        return facts
    
    def _extract_clause_references(self, content: str) -> List[str]:
        """Extract policy clause references"""
        patterns = [
            r'clause\s+(\d+(?:\.\d+)*)',
            r'section\s+(\d+(?:\.\d+)*)',
            r'paragraph\s+(\d+(?:\.\d+)*)',
            r'article\s+(\d+(?:\.\d+)*)',
            r'sub-section\s+(\d+(?:\.\d+)*)'
        ]
        
        clauses = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            clauses.extend([f"Clause {match}" for match in matches])
        
        return list(set(clauses))  # Remove duplicates
    
    def _generate_concise_answer(self, content: str, facts: List[ExtractedFact], 
                                query_type: Optional[str] = None) -> str:
        """Generate concise answer from content and facts"""
        
        # Identify answer type based on facts and query
        if any(f.type == FactType.TIME_PERIOD for f in facts):
            if 'grace' in content.lower():
                time_fact = next(f for f in facts if f.type == FactType.TIME_PERIOD)
                return f"A grace period of {time_fact.value} is provided for premium payment after the due date."
            elif 'waiting' in content.lower():
                time_fact = next(f for f in facts if f.type == FactType.TIME_PERIOD)
                return f"There is a waiting period of {time_fact.value} of continuous coverage."
        
        # For monetary questions
        if any(f.type == FactType.MONETARY_AMOUNT for f in facts):
            money_fact = next(f for f in facts if f.type == FactType.MONETARY_AMOUNT)
            if 'premium' in content.lower():
                return f"The premium amount is {money_fact.value}."
            elif 'deductible' in content.lower():
                return f"The deductible is {money_fact.value}."
        
        # For coverage questions
        if 'cover' in content.lower() or 'benefit' in content.lower():
            # Extract key coverage information
            coverage_sentence = self._extract_key_sentence(content, ['cover', 'benefit', 'include'])
            if coverage_sentence:
                return self._clean_sentence(coverage_sentence)
        
        # For exclusion questions
        if 'exclud' in content.lower() or 'not cover' in content.lower():
            exclusion_sentence = self._extract_key_sentence(content, ['exclud', 'not cover', 'except'])
            if exclusion_sentence:
                return self._clean_sentence(exclusion_sentence)
        
        # Default: use first complete sentence
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if len(sentence.strip()) > 20 and not sentence.strip().startswith(('Note:', 'However,')):
                return self._clean_sentence(sentence.strip()) + "."
        
        # Final fallback
        return self._clean_verbose_response(content)[:200] + "..."
    
    def _extract_key_sentence(self, content: str, keywords: List[str]) -> Optional[str]:
        """Extract the most relevant sentence containing keywords"""
        sentences = re.split(r'[.!?]+', content)
        
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            score = sum(1 for keyword in keywords if keyword in sentence.lower())
            if score > 0:
                scored_sentences.append((sentence, score))
        
        if scored_sentences:
            # Return sentence with highest score
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            return scored_sentences[0][0]
        
        return None
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean and normalize a sentence"""
        # Remove JSON artifacts
        sentence = re.sub(r'[{}"\[\]]', '', sentence)
        
        # Remove redundant phrases
        redundant_phrases = [
            r'according to the policy',
            r'as per the document',
            r'the policy states that',
            r'it is mentioned that',
            r'the document indicates',
        ]
        
        for phrase in redundant_phrases:
            sentence = re.sub(phrase, '', sentence, flags=re.IGNORECASE)
        
        # Clean up whitespace
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        return sentence
    
    def _clean_verbose_response(self, content: str) -> str:
        """Clean verbose response by removing unnecessary parts"""
        # Remove common verbose patterns
        verbose_patterns = [
            r'Based on the policy document,?\s*',
            r'According to the insurance policy,?\s*',
            r'The policy clearly states that\s*',
            r'It should be noted that\s*',
            r'Please be aware that\s*',
            r'In summary,?\s*',
            r'To conclude,?\s*'
        ]
        
        cleaned = content
        for pattern in verbose_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Take first paragraph or sentence
        paragraphs = cleaned.split('\n\n')
        if paragraphs:
            first_para = paragraphs[0].strip()
            if len(first_para) > 0:
                return first_para
        
        return cleaned.strip()
    
    def _get_sentence_context(self, content: str, position: int, window: int = 100) -> str:
        """Get sentence context around a position"""
        start = max(0, position - window)
        end = min(len(content), position + window)
        
        context = content[start:end]
        
        # Try to get complete sentences
        sentences = re.split(r'[.!?]+', context)
        if len(sentences) >= 2:
            return sentences[1].strip()  # Middle sentence
        
        return context.strip()
    
    def _extract_reasoning(self, response_data: Dict, content: str) -> List[str]:
        """Extract reasoning steps from response"""
        reasoning = []
        
        # From structured response
        if response_data:
            reasoning_fields = ['reasoning_steps', 'reasoning_chain', 'reasoning']
            for field in reasoning_fields:
                if field in response_data:
                    steps = response_data[field]
                    if isinstance(steps, list):
                        reasoning.extend(steps)
                    elif isinstance(steps, str):
                        reasoning.append(steps)
        
        # From text patterns
        if not reasoning:
            reasoning_patterns = [
                r'because\s+([^.]+)',
                r'due to\s+([^.]+)',
                r'since\s+([^.]+)',
                r'as\s+([^.]+)'
            ]
            
            for pattern in reasoning_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                reasoning.extend(matches[:2])  # Limit to 2 per pattern
        
        return reasoning[:5]  # Limit to top 5 reasoning points
    
    def _calculate_optimization_confidence(self, facts: List[ExtractedFact], 
                                         clauses: List[str], content: str) -> float:
        """Calculate confidence in the optimization"""
        base_confidence = 0.5
        
        # Boost for extracted facts
        if facts:
            fact_boost = min(0.3, len(facts) * 0.05)
            base_confidence += fact_boost
        
        # Boost for clause references
        if clauses:
            clause_boost = min(0.2, len(clauses) * 0.05)
            base_confidence += clause_boost
        
        # Boost for clear structure
        if any(keyword in content.lower() for keyword in ['grace period', 'waiting period', 'coverage', 'premium']):
            base_confidence += 0.2
        
        # Penalty for vague content
        if any(phrase in content.lower() for phrase in ['may', 'might', 'possibly', 'unclear']):
            base_confidence -= 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def _generate_fallback_answer(self, content: str, facts: List[ExtractedFact]) -> str:
        """Generate fallback answer when confidence is low"""
        if facts:
            fact_summaries = []
            for fact in facts[:3]:  # Top 3 facts
                fact_summaries.append(f"{fact.type.value}: {fact.value}")
            return "Key information: " + "; ".join(fact_summaries)
        
        # Use first meaningful sentence
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if len(sentence.strip()) > 15:
                return sentence.strip() + "."
        
        return "Please refer to the specific policy clauses for detailed information."
    
    def _create_fallback_optimization(self, raw_response: str, error_msg: str) -> OptimizedResponse:
        """Create fallback optimization when processing fails"""
        logger.warning(f"[kg290] Creating fallback optimization: {error_msg}")
        
        return OptimizedResponse(
            concise_answer="Unable to optimize response. Please refer to the original answer.",
            key_facts=[],
            clause_references=[],
            confidence=0.3,
            reasoning=[f"Optimization failed: {error_msg}"],
            fallback_answer=raw_response[:200] + "..." if len(raw_response) > 200 else raw_response,
            original_response=raw_response
        )
    
    def format_for_api(self, optimized: OptimizedResponse, include_details: bool = False) -> Dict[str, Any]:
        """Format optimized response for API output"""
        result = {
            "answer": optimized.concise_answer,
            "confidence": optimized.confidence
        }
        
        if include_details:
            result.update({
                "key_facts": [
                    {
                        "type": fact.type.value,
                        "value": fact.value,
                        "confidence": fact.confidence
                    } for fact in optimized.key_facts
                ],
                "clause_references": optimized.clause_references,
                "reasoning": optimized.reasoning
            })
            
            if optimized.fallback_answer:
                result["fallback_answer"] = optimized.fallback_answer
        
        return result


# Factory function
def create_response_optimizer() -> ResponseOptimizer:
    """Factory function to create response optimizer"""
    logger.info("[kg290] Creating ResponseOptimizer instance")
    return ResponseOptimizer()


# Utility functions for common optimizations
def optimize_for_hackathon(raw_response: str, query: str = "") -> str:
    """Quick optimization for hackathon evaluation criteria"""
    optimizer = create_response_optimizer()
    optimized = optimizer.optimize_response(raw_response, style=ResponseStyle.CONCISE)
    return optimized.concise_answer


def extract_policy_facts(content: str) -> List[Dict[str, Any]]:
    """Extract policy facts for analysis"""
    optimizer = create_response_optimizer()
    facts = optimizer._extract_all_facts(content)
    
    return [
        {
            "type": fact.type.value,
            "value": fact.value,
            "context": fact.context[:100] + "..." if len(fact.context) > 100 else fact.context,
            "confidence": fact.confidence
        } for fact in facts
    ]