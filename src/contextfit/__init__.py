"""
ContextFit: Token-native knowledge base for LLM scale.

Everything stays in token space until final generation.
"""

__version__ = "0.1.1"

from contextfit.core.chunk import Chunk, ChunkStore
from contextfit.core.tokenizer import Tokenizer
from contextfit.graph.community import CommunityDetector
from contextfit.graph.similarity import LSHIndex, MinHasher
from contextfit.hierarchy.levels import HierarchyBuilder
from contextfit.index.bm25 import BM25Scorer
from contextfit.index.inverted import InvertedIndex
from contextfit.retrieval.engine import RetrievalEngine
from contextfit.sid.generator import SIDGenerator, SIDPrediction
from contextfit.sid.learned import LearnedSIDGenerator
from contextfit.sid.semantic import SemanticID, SemanticIDIndex

__all__ = [
    "Chunk",
    "ChunkStore",
    "Tokenizer",
    "LearnedSIDGenerator",
    "InvertedIndex",
    "BM25Scorer",
    "MinHasher",
    "LSHIndex",
    "CommunityDetector",
    "HierarchyBuilder",
    "RetrievalEngine",
    "SIDGenerator",
    "SIDPrediction",
    "SemanticID",
    "SemanticIDIndex",
]
