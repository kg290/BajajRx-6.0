"""
Enhanced Embedder module with optimized chunking for Llama 4 Scout
Author: kg290
Date: 2025-07-26 11:40:07
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyEmbedder:
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        """
        Initialize embedder optimized for Llama 4 Scout's 128K context window
        """
        logger.info(f"[kg290] Initializing PolicyEmbedder with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"[kg290] Embedding dimension: {self.embedding_dim}")

    def chunk_text_with_metadata(self, text: str, chunk_size: int = 1200, overlap: int = 100) -> Tuple[
        List[str], List[Dict]]:
        """
        Enhanced chunking strategy optimized for Llama 4 Scout's capabilities

        Args:
            text: Raw text to chunk
            chunk_size: Size of each chunk (increased for Llama 4)
            overlap: Overlap between chunks for context continuity

        Returns:
            Tuple of (chunks, metadata)
        """
        logger.info(f"[kg290] Chunking text with size={chunk_size}, overlap={overlap}")

        chunks = []
        metadata = []

        # Smart chunking: try to break at sentence boundaries
        sentences = text.split('. ')
        current_chunk = ""
        start_pos = 0

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip() + '. '

            if len(current_chunk + sentence) <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    metadata.append({
                        "chunk_id": len(chunks) - 1,
                        "start_pos": start_pos,
                        "end_pos": start_pos + len(current_chunk),
                        "length": len(current_chunk),
                        "sentence_count": current_chunk.count('.'),
                        "timestamp": "2025-07-26 11:40:07",
                        "user": "kg290"
                    })

                # Start new chunk with overlap
                if overlap > 0 and chunks:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + sentence
                    start_pos = start_pos + len(chunks[-1]) - overlap
                else:
                    current_chunk = sentence
                    start_pos = start_pos + len(chunks[-1]) if chunks else 0

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            metadata.append({
                "chunk_id": len(chunks) - 1,
                "start_pos": start_pos,
                "end_pos": start_pos + len(current_chunk),
                "length": len(current_chunk),
                "sentence_count": current_chunk.count('.'),
                "timestamp": "2025-07-26 11:40:07",
                "user": "kg290"
            })

        logger.info(f"[kg290] Created {len(chunks)} chunks with enhanced metadata")
        return chunks, metadata

    def create_faiss_index(self, chunks: List[str]) -> Tuple[faiss.Index, np.ndarray]:
        """
        Create optimized FAISS index for fast retrieval

        Args:
            chunks: List of text chunks

        Returns:
            Tuple of (FAISS index, embeddings array)
        """
        logger.info(f"[kg290] Creating FAISS index for {len(chunks)} chunks")

        # Generate embeddings with batch processing for efficiency
        embeddings = self.model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Create optimized FAISS index
        index = faiss.IndexFlatL2(self.embedding_dim)

        # Add embeddings to index
        embeddings_float32 = embeddings.astype('float32')
        index.add(embeddings_float32)

        logger.info(f"[kg290] FAISS index created successfully with {index.ntotal} vectors")
        return index, embeddings_float32

    def create_hnsw_index(self, chunks: List[str], M: int = 32, ef_construction: int = 200) -> Tuple[
        faiss.Index, np.ndarray]:
        """
        Create HNSW index for faster approximate search (for large datasets)

        Args:
            chunks: List of text chunks
            M: HNSW parameter (connections per node)
            ef_construction: HNSW parameter (search width during construction)

        Returns:
            Tuple of (HNSW index, embeddings array)
        """
        logger.info(f"[kg290] Creating HNSW index for {len(chunks)} chunks with M={M}, ef={ef_construction}")

        embeddings = self.model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Create HNSW index for faster search
        index = faiss.IndexHNSWFlat(self.embedding_dim, M)
        index.hnsw.ef_construction = ef_construction

        embeddings_float32 = embeddings.astype('float32')
        index.add(embeddings_float32)

        logger.info(f"[kg290] HNSW index created with {index.ntotal} vectors")
        return index, embeddings_float32

    def get_embedding_stats(self, embeddings: np.ndarray) -> Dict:
        """
        Get statistics about embeddings for debugging
        """
        return {
            "shape": embeddings.shape,
            "mean": float(np.mean(embeddings)),
            "std": float(np.std(embeddings)),
            "min": float(np.min(embeddings)),
            "max": float(np.max(embeddings)),
            "user": "kg290",
            "timestamp": "2025-07-26 11:40:07"
        }


# Factory function for easy initialization
def create_embedder(model_name: str = "intfloat/e5-base-v2") -> PolicyEmbedder:
    """Factory function to create embedder instance"""
    logger.info(f"[kg290] Creating PolicyEmbedder instance")
    return PolicyEmbedder(model_name)