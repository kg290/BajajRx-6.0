"""
Enhanced Retrieval module optimized for Llama 4 Scout's 128K context
Author: kg290
Date: 2025-07-26 11:40:07
"""

import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: List[str]
    metadata: List[Dict]
    scores: List[float]
    query_embedding: np.ndarray
    total_retrieved: int
    retrieval_strategy: str


@dataclass
class RetrievalConfig:
    top_k: int = 10  # Increased for Llama 4 Scout's larger context
    score_threshold: float = 0.7
    max_context_length: int = 120000  # Utilize 128K context window
    rerank: bool = True
    diversity_penalty: float = 0.1


class EnhancedRetriever:
    def __init__(self, model: SentenceTransformer, index: faiss.Index,
                 chunks: List[str], metadata: List[Dict]):
        """
        Initialize enhanced retriever optimized for Llama 4 Scout
        """
        self.model = model
        self.index = index
        self.chunks = chunks
        self.metadata = metadata
        self.config = RetrievalConfig()

        logger.info(f"[kg290] EnhancedRetriever initialized with {len(chunks)} chunks")
        logger.info(f"[kg290] Index type: {type(index).__name__}")

    def retrieve_with_scores(self, query: str, config: Optional[RetrievalConfig] = None) -> RetrievalResult:
        """
        Enhanced retrieval with scoring and filtering optimized for Llama 4 Scout
        """
        if config is None:
            config = self.config

        logger.info(f"[kg290] Retrieving with top_k={config.top_k}, threshold={config.score_threshold}")

        # Generate query embedding
        query_embedding = self.model.encode([query], show_progress_bar=False)

        # Search index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(config.top_k * 2, len(self.chunks))  # Get more candidates for filtering
        )

        # Convert distances to similarity scores (L2 distance -> similarity)
        similarities = 1 / (1 + distances[0])

        # Filter by score threshold
        filtered_results = []
        for idx, score in zip(indices[0], similarities):
            if score >= config.score_threshold and idx < len(self.chunks):
                filtered_results.append((idx, score))

        # If not enough results, lower threshold
        if len(filtered_results) < config.top_k // 2:
            logger.warning(f"[kg290] Low results with threshold {config.score_threshold}, relaxing...")
            filtered_results = [(idx, score) for idx, score in zip(indices[0], similarities)
                                if idx < len(self.chunks)]

        # Apply diversity penalty if enabled
        if config.rerank and len(filtered_results) > config.top_k:
            filtered_results = self._apply_diversity_penalty(filtered_results, config.diversity_penalty)

        # Take top K results
        final_results = filtered_results[:config.top_k]

        # Extract chunks and metadata
        retrieved_chunks = []
        retrieved_metadata = []
        scores = []

        for idx, score in final_results:
            retrieved_chunks.append(self.chunks[idx])
            retrieved_metadata.append({
                **self.metadata[idx],
                "retrieval_score": float(score),
                "retrieval_rank": len(scores) + 1,
                "user": "kg290",
                "timestamp": "2025-07-26 11:40:07"
            })
            scores.append(score)

        # Optimize context length for Llama 4 Scout
        optimized_chunks, optimized_metadata = self._optimize_context_length(
            retrieved_chunks, retrieved_metadata, config.max_context_length
        )

        result = RetrievalResult(
            chunks=optimized_chunks,
            metadata=optimized_metadata,
            scores=scores[:len(optimized_chunks)],
            query_embedding=query_embedding[0],
            total_retrieved=len(optimized_chunks),
            retrieval_strategy="enhanced_with_scoring"
        )

        logger.info(
            f"[kg290] Retrieved {len(optimized_chunks)} chunks, total chars: {sum(len(c) for c in optimized_chunks)}")
        return result

    def _apply_diversity_penalty(self, results: List[Tuple[int, float]], penalty: float) -> List[Tuple[int, float]]:
        """
        Apply diversity penalty to avoid too similar chunks
        """
        if penalty <= 0:
            return results

        logger.debug(f"[kg290] Applying diversity penalty: {penalty}")

        selected = []
        remaining = results.copy()

        # Always take the top result
        if remaining:
            selected.append(remaining.pop(0))

        while remaining and len(selected) < len(results):
            best_idx = 0
            best_score = -1

            for i, (chunk_idx, score) in enumerate(remaining):
                # Calculate diversity penalty
                diversity_bonus = 1.0
                for selected_idx, _ in selected:
                    # Simple diversity based on chunk position distance
                    position_distance = abs(chunk_idx - selected_idx)
                    diversity_bonus *= (1 + penalty * min(position_distance / 10, 1))

                adjusted_score = score * diversity_bonus

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _optimize_context_length(self, chunks: List[str], metadata: List[Dict],
                                 max_length: int) -> Tuple[List[str], List[Dict]]:
        """
        Optimize context length for Llama 4 Scout's 128K window
        """
        total_length = sum(len(chunk) for chunk in chunks)

        if total_length <= max_length:
            logger.debug(f"[kg290] Context within limits: {total_length}/{max_length}")
            return chunks, metadata

        logger.info(f"[kg290] Optimizing context: {total_length} -> {max_length}")

        # Progressive truncation strategy
        optimized_chunks = []
        optimized_metadata = []
        current_length = 0

        for chunk, meta in zip(chunks, metadata):
            if current_length + len(chunk) <= max_length:
                optimized_chunks.append(chunk)
                optimized_metadata.append(meta)
                current_length += len(chunk)
            else:
                # Try to fit partial chunk
                remaining_space = max_length - current_length
                if remaining_space > 200:  # Minimum meaningful chunk size
                    truncated_chunk = chunk[:remaining_space - 50] + "..."
                    optimized_chunks.append(truncated_chunk)
                    optimized_metadata.append({
                        **meta,
                        "truncated": True,
                        "original_length": len(chunk),
                        "truncated_length": len(truncated_chunk)
                    })
                break

        logger.info(f"[kg290] Context optimized: {len(optimized_chunks)} chunks, {current_length} chars")
        return optimized_chunks, optimized_metadata

    def hybrid_retrieve(self, query: str, keywords: List[str],
                        config: Optional[RetrievalConfig] = None) -> RetrievalResult:
        """
        Hybrid retrieval combining semantic and keyword-based search
        """
        if config is None:
            config = self.config

        logger.info(f"[kg290] Hybrid retrieval for query with keywords: {keywords}")

        # Semantic retrieval
        semantic_result = self.retrieve_with_scores(query, config)

        # Keyword-based filtering
        keyword_boosted_chunks = []
        keyword_boosted_metadata = []
        keyword_boosted_scores = []

        for chunk, meta, score in zip(semantic_result.chunks, semantic_result.metadata, semantic_result.scores):
            # Calculate keyword boost
            keyword_count = sum(1 for keyword in keywords if keyword.lower() in chunk.lower())
            keyword_boost = 1 + (keyword_count * 0.1)  # 10% boost per keyword match

            boosted_score = score * keyword_boost

            keyword_boosted_chunks.append(chunk)
            keyword_boosted_metadata.append({
                **meta,
                "keyword_matches": keyword_count,
                "keyword_boost": keyword_boost,
                "boosted_score": boosted_score
            })
            keyword_boosted_scores.append(boosted_score)

        # Re-sort by boosted scores
        sorted_indices = sorted(range(len(keyword_boosted_scores)),
                                key=lambda i: keyword_boosted_scores[i], reverse=True)

        final_chunks = [keyword_boosted_chunks[i] for i in sorted_indices]
        final_metadata = [keyword_boosted_metadata[i] for i in sorted_indices]
        final_scores = [keyword_boosted_scores[i] for i in sorted_indices]

        return RetrievalResult(
            chunks=final_chunks,
            metadata=final_metadata,
            scores=final_scores,
            query_embedding=semantic_result.query_embedding,
            total_retrieved=len(final_chunks),
            retrieval_strategy="hybrid_semantic_keyword"
        )

    def get_retrieval_stats(self) -> Dict:
        """
        Get retrieval system statistics
        """
        return {
            "total_chunks": len(self.chunks),
            "index_type": type(self.index).__name__,
            "embedding_dim": self.model.get_sentence_embedding_dimension(),
            "config": {
                "top_k": self.config.top_k,
                "score_threshold": self.config.score_threshold,
                "max_context_length": self.config.max_context_length,
                "rerank": self.config.rerank
            },
            "user": "kg290",
            "timestamp": "2025-07-26 11:40:07"
        }


# Factory function
def create_retriever(model: SentenceTransformer, index: faiss.Index,
                     chunks: List[str], metadata: List[Dict],
                     config: Optional[RetrievalConfig] = None) -> EnhancedRetriever:
    """Factory function to create retriever instance"""
    logger.info(f"[kg290] Creating EnhancedRetriever instance")
    retriever = EnhancedRetriever(model, index, chunks, metadata)
    if config:
        retriever.config = config
    return retriever