"""
Mihaji Memory Store — ChromaDB-backed vector memory for Hermes Agent.

Adapted from mihaji-memory/scripts/mem.py, ported to macOS / Hermes MemoryProvider.

Features:
  - Persistent ChromaDB with sentence-transformers multilingual embeddings
  - Overlapping chunking (300 chars, 50 overlap) for context coherence
  - Rich metadata: timestamps, group IDs, chunk indices
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# Default model — lightweight multilingual, auto-downloaded by sentence-transformers
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Chunking constants (from original mem.py)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

COLLECTION_NAME = "mihaji_multilingual_memory"


class MihajiStore:
    """ChromaDB-backed vector memory store for mihaji-memory plugin."""

    def __init__(
        self,
        db_path: str,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        """
        Args:
            db_path: Directory where ChromaDB persists its data.
            model_name: sentence-transformers model name or local path.
        """
        self._db_path = Path(db_path)
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._embedding_fn = None  # Lazy-loaded on first use

        self._client = chromadb.PersistentClient(path=str(self._db_path))
        # Create collection without embedding function — we set it lazily
        # to avoid the ~15s sentence-transformers model load at startup.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
        )

    def _get_embedding_fn(self):
        """Lazy-load the sentence-transformers model.

        The first call loads the model (~15s for 199-layer multilingual),
        subsequent calls are instant. Called automatically by add/search.
        """
        if self._embedding_fn is None:
            logger.info("Mihaji: loading embedding model '%s' (first time ~15s)...", self._model_name)
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._model_name,
            )
            self._collection.embedding_function = self._embedding_fn
            logger.info("Mihaji: embedding model loaded")
        return self._embedding_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a text to memory, auto-chunked with overlapping windows.

        Returns the number of chunks stored.
        """
        self._get_embedding_fn()  # Lazy-load model on first use
        try:
            chunks = _chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

            group_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{group_id}_{i}"
                ids.append(chunk_id)
                documents.append(chunk)

                meta = dict(metadata or {}, source="mihaji-memory")
                meta.update(
                    created_at=timestamp,
                    group_id=group_id,
                    chunk_idx=i,
                    total_chunks=len(chunks),
                )
                metadatas.append(meta)

            self._collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )
            logger.debug("Mihaji: stored %d chunks (group %s)", len(chunks), group_id)
            return len(chunks)

        except Exception:
            logger.exception("Mihaji add failed")
            return 0

    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Semantic search across stored memories.

        Returns a list of dicts with: content, created_at, chunk_idx, total_chunks, group_id.
        """
        self._get_embedding_fn()  # Lazy-load model on first use
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
            )

            documents: List[str] = results["documents"][0]
            metas: List[Dict[str, Any]] = results["metadatas"][0]

            if not documents:
                return []

            memories: List[Dict[str, Any]] = []
            for doc, meta in zip(documents, metas):
                memories.append(
                    {
                        "content": doc,
                        "created_at": meta.get("created_at", ""),
                        "chunk_idx": meta.get("chunk_idx", 0),
                        "total_chunks": meta.get("total_chunks", 1),
                        "group_id": meta.get("group_id", ""),
                    }
                )
            return memories

        except Exception:
            logger.exception("Mihaji search failed")
            return []

    def count(self) -> int:
        """Return total number of stored chunks."""
        try:
            return self._collection.count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def model_name(self) -> str:
        return self._model_name


# ------------------------------------------------------------------
# Chunking (from original mem.py)
# ------------------------------------------------------------------

def _chunk_text(text: str, size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for context coherence."""
    if len(text) <= size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap
        if end >= len(text):
            break
    return chunks
