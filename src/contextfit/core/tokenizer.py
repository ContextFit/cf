"""
Tokenizer wrapper: Unified interface for various tokenizers.

Supports tiktoken (OpenAI), sentencepiece, and potentially HuggingFace tokenizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class BaseTokenizer(ABC):
    """Abstract base for tokenizers."""
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return tokenizer name/identifier."""
        ...
    
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode text to token IDs."""
        ...
    
    @abstractmethod
    def decode(self, tokens: np.ndarray | list[int]) -> str:
        """Decode token IDs to text."""
        ...
    
    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts."""
        return [self.encode(t) for t in texts]
    
    def decode_batch(self, token_lists: list[np.ndarray | list[int]]) -> list[str]:
        """Decode multiple token sequences."""
        return [self.decode(t) for t in token_lists]


class TiktokenWrapper(BaseTokenizer):
    """Wrapper for tiktoken (OpenAI tokenizers)."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._name = encoding_name
    
    @property
    def vocab_size(self) -> int:
        return self._encoding.n_vocab
    
    @property
    def name(self) -> str:
        return f"tiktoken:{self._name}"
    
    def encode(self, text: str) -> np.ndarray:
        # Disable special token checks to handle arbitrary text (e.g., session logs)
        tokens = self._encoding.encode(text, disallowed_special=())
        return np.array(tokens, dtype=np.uint32)
    
    def decode(self, tokens: np.ndarray | list[int]) -> str:
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return self._encoding.decode(tokens)


class Tokenizer:
    """
    Factory and unified interface for tokenizers.
    
    Usage:
        tokenizer = Tokenizer.load("tiktoken:cl100k_base")
        tokens = tokenizer.encode("Hello world")
        text = tokenizer.decode(tokens)
    """
    
    _cache: dict[str, BaseTokenizer] = {}
    
    def __init__(self, backend: BaseTokenizer):
        self._backend = backend
    
    @classmethod
    def load(
        cls,
        name: str | Literal["tiktoken", "cl100k_base", "o200k_base"] = "cl100k_base",
    ) -> Tokenizer:
        """
        Load a tokenizer by name.
        
        Supported formats:
            - "cl100k_base" (tiktoken, GPT-4/3.5)
            - "o200k_base" (tiktoken, GPT-4o)
            - "tiktoken:encoding_name"
        """
        # Normalize name
        if name in ("tiktoken", "cl100k_base"):
            key = "tiktoken:cl100k_base"
        elif name == "o200k_base":
            key = "tiktoken:o200k_base"
        elif name.startswith("tiktoken:"):
            key = name
        else:
            key = f"tiktoken:{name}"
        
        # Check cache
        if key not in cls._cache:
            encoding = key.split(":", 1)[1]
            cls._cache[key] = TiktokenWrapper(encoding)
        
        return cls(cls._cache[key])
    
    @property
    def vocab_size(self) -> int:
        return self._backend.vocab_size
    
    @property
    def name(self) -> str:
        return self._backend.name
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to token IDs (uint32 array)."""
        return self._backend.encode(text)
    
    def decode(self, tokens: np.ndarray | list[int]) -> str:
        """Decode token IDs to text."""
        return self._backend.decode(tokens)
    
    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts."""
        return self._backend.encode_batch(texts)
    
    def decode_batch(self, token_lists: list[np.ndarray | list[int]]) -> list[str]:
        """Decode multiple token sequences."""
        return self._backend.decode_batch(token_lists)
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[np.ndarray]:
        """
        Chunk text into overlapping token sequences.
        
        Args:
            text: Input text
            chunk_size: Target tokens per chunk
            overlap: Token overlap between chunks
        
        Returns:
            List of token arrays
        """
        tokens = self.encode(text)
        
        if len(tokens) <= chunk_size:
            return [tokens]
        
        chunks = []
        start = 0
        step = chunk_size - overlap
        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunks.append(tokens[start:end])
            
            if end >= len(tokens):
                break
            start += step
        
        return chunks
