"""
Mihaji Memory Store — ChromaDB-backed vector memory with BM25 hybrid search.

Adapted from mihaji-memory/scripts/mem.py, ported to macOS / Hermes MemoryProvider.

Features:
  - Persistent ChromaDB with sentence-transformers multilingual embeddings
  - Overlapping chunking (300 chars, 50 overlap) for context coherence
  - Rich metadata: timestamps, group IDs, chunk indices
  - Hybrid search: BM25 (jieba keyword) + Vector (semantic) via RRF fusion
  - Lazy BM25 index — auto-rebuilt when stale
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

try:
    # When imported as part of the mihaji package
    from .hybrid_search import BM25Index, build_index_from_collection, rrf_fuse, weighted_sample
except ImportError:
    # When run directly as a script or test
    from hybrid_search import BM25Index, build_index_from_collection, rrf_fuse, weighted_sample  # type: ignore[no-redef]

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model — lightweight multilingual, auto-downloaded by sentence-transformers
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Chunking constants (from original mem.py)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

COLLECTION_NAME = "mihaji_multilingual_memory"

# How many extra candidates to pull from each leg for fusion
HYBRID_VECTOR_CANDIDATES = 20  # Fetch 20 from vector to give BM25 room to boost
HYBRID_BM25_CANDIDATES = 20


def _resolve_model_path(model_name: str) -> tuple[str, bool]:
    """Resolve model_name to local snapshot directory if cached, returning (resolved_path, is_local)."""
    p = Path(model_name)
    if p.exists() and p.is_dir():
        return str(p), True

    # Check ~/.cache/huggingface/hub/models--...
    hub_name = "models--" + model_name.replace("/", "--")
    if not hub_name.startswith("models--sentence-transformers--") and "/" not in model_name:
        hub_name = "models--sentence-transformers--" + model_name

    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / hub_name / "snapshots"
    if snapshots.exists():
        for snap in snapshots.iterdir():
            if snap.is_dir() and ((snap / "config.json").exists() or (snap / "model.safetensors").exists()):
                return str(snap), True
    return model_name, False


class MihajiStore:
    """ChromaDB-backed vector memory store with optional BM25 hybrid search."""

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
        self._model: Optional[SentenceTransformer] = None
        self._model_loading = False
        self._model_lock = threading.Lock()
        self._bm25: Optional[BM25Index] = None  # Lazy-built on first hybrid search

        # Disable ChromaDB telemetry to prevent any OpenTelemetry collision
        chroma_settings = chromadb.Settings(anonymized_telemetry=False)
        self._client = chromadb.PersistentClient(path=str(self._db_path), settings=chroma_settings)
        self._bm25_cache_file = self._db_path / "bm25_cache.json"
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
        )

        # Trigger non-blocking background model warmup
        self.warmup()

    # ------------------------------------------------------------------
    # Embedding (background warmup & lazy fallback)
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Start non-blocking background model loading thread."""
        if self._model is not None or self._model_loading:
            return
        t = threading.Thread(target=self._load_model_worker, daemon=True, name="mihaji-warmup")
        t.start()

    def is_model_ready(self) -> bool:
        """Check if embedding model is loaded and ready."""
        return self._model is not None

    def _load_model_worker(self) -> None:
        """Worker thread to load sentence-transformers model directly from local files."""
        with self._model_lock:
            if self._model is not None:
                return
            self._model_loading = True
            try:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                model_path, is_local = _resolve_model_path(self._model_name)
                logger.info("Mihaji: loading embedding model from '%s' (is_local=%s)...", model_path, is_local)

                if is_local:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    self._model = SentenceTransformer(
                        model_path,
                        local_files_only=True,
                    )
                else:
                    # Online mirror fallback
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                    self._model = SentenceTransformer(
                        model_path,
                    )
                logger.info("Mihaji: embedding model loaded and ready")
            except Exception as e:
                logger.error("Mihaji: failed to load embedding model: %s", e)
            finally:
                self._model_loading = False

    def _encode(self, texts: List[str], block: bool = True) -> Optional[List[List[float]]]:
        """Encode texts into embedding vectors directly via SentenceTransformer."""
        if self._model is None:
            if block:
                self._load_model_worker()
            else:
                return None
        if self._model is None:
            return None
        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=False)
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # BM25 index (cached + incremental)
    # ------------------------------------------------------------------

    def _ensure_bm25(self) -> BM25Index:
        """Lazy-build or load the BM25 index from disk cache / collection."""
        if self._bm25 is None:
            self._bm25 = BM25Index()
            total_count = self.count()
            # Try fast disk cache load first
            if total_count > 0 and self._bm25.load_cache(self._bm25_cache_file, expected_count=total_count):
                return self._bm25
            # Cache miss or count mismatch: rebuild from collection and cache
            self._bm25 = build_index_from_collection(self._collection)
            self._bm25.save_cache(self._bm25_cache_file)
        elif self._bm25.needs_rebuild:
            self._bm25 = build_index_from_collection(self._collection)
            self._bm25.save_cache(self._bm25_cache_file)
        return self._bm25

    def _mark_bm25_stale(self):
        """Call after batch mutations to trigger lazy BM25 rebuild."""
        if self._bm25 is not None:
            self._bm25.mark_stale()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        strength: int = 50,
        memory_type: str = "general",
        tags: Optional[List[str]] = None,
    ) -> int:
        """Add a text to memory, auto-chunked with overlapping windows.

        Args:
            text: The text content to memorize.
            metadata: Optional extra metadata dict.
            strength: Importance 1-100 (higher = more likely in weighted recall).
            memory_type: Category like "knowledge", "preference", "event", "task", "general".
            tags: Keyword tags for filtering (stored as comma-separated string).

        Returns:
            Number of chunks stored.
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

                # Normalize tags: list → comma-separated string
                tags_str = ""
                if tags:
                    if isinstance(tags, str):
                        tags_str = tags
                    else:
                        tags_str = ",".join(str(t).strip() for t in tags if t)

                meta = dict(metadata or {}, source="mihaji-memory")
                meta.update(
                    created_at=timestamp,
                    group_id=group_id,
                    chunk_idx=i,
                    total_chunks=len(chunks),
                    strength=strength,
                    memory_type=memory_type,
                    tags=tags_str,
                )
                metadatas.append(meta)

            # Compute embeddings directly via local SentenceTransformer model
            embs = self._encode(documents, block=True)
            if embs:
                self._collection.add(
                    embeddings=embs,
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas,
                )
            else:
                self._collection.add(
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas,
                )

            # Incrementally update BM25 index and disk cache if already loaded
            if self._bm25 is not None and not self._bm25.needs_rebuild:
                self._bm25.add_documents(documents, new_doc_ids=ids, new_doc_metas=metadatas)
                self._bm25.save_cache(self._bm25_cache_file)
            else:
                self._mark_bm25_stale()

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
        """Pure vector search (backward-compatible).

        Returns a list of dicts with: content, created_at, chunk_idx, total_chunks, group_id.
        """
        try:
            q_emb = self._encode([query], block=True)
            if q_emb:
                results = self._collection.query(
                    query_embeddings=q_emb,
                    n_results=n_results,
                )
            else:
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
                        "strength": meta.get("strength", 50),
                        "memory_type": meta.get("memory_type", "general"),
                        "tags": _parse_tags(meta.get("tags", "")),
                    }
                )
            return memories

        except Exception:
            logger.exception("Mihaji search failed")
            return []

    def search_hybrid(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: BM25 (keyword) + Vector (semantic) fused via RRF.

        Returns results in the same format as search() for drop-in replacement:
        {content, created_at, chunk_idx, total_chunks, group_id} + rrf_score.

        Graceful degradation:
          - Vector still warming up / unavailable → instant pure BM25 results (<2ms)
          - BM25 unavailable → pure vector results
        """
        # -- Leg 1: Vector search (pull extra candidates if model is ready) --
        vector_results: List[Dict[str, Any]] = []
        if self.is_model_ready():
            try:
                q_emb = self._encode([query], block=False)
                if q_emb:
                    raw = self._collection.query(
                        query_embeddings=q_emb,
                        n_results=HYBRID_VECTOR_CANDIDATES,
                    )
                    docs: List[str] = raw["documents"][0]
                    metas: List[Dict[str, Any]] = raw["metadatas"][0]
                    for doc, meta in zip(docs, metas):
                        vector_results.append({
                            "content": doc,
                            "created_at": meta.get("created_at", ""),
                            "chunk_idx": meta.get("chunk_idx", 0),
                            "total_chunks": meta.get("total_chunks", 1),
                            "group_id": meta.get("group_id", ""),
                            "strength": meta.get("strength", 50),
                            "memory_type": meta.get("memory_type", "general"),
                            "tags": _parse_tags(meta.get("tags", "")),
                        })
            except Exception as e:
                logger.debug("Vector search failed (BM25 will cover): %s", e)
        else:
            # Trigger background warmup if not started yet; do not block this prefetch turn
            self.warmup()
            logger.debug("Mihaji: vector model warming up in background; serving instant BM25 leg")

        # -- Leg 2: BM25 keyword search --
        bm25_results: List[Dict[str, Any]] = []
        try:
            bm25 = self._ensure_bm25()
            bm25_results = bm25.search(query, top_k=HYBRID_BM25_CANDIDATES)
        except Exception as e:
            logger.debug("BM25 search failed (vector will cover): %s", e)

        # -- Degradation checks --
        if not vector_results and not bm25_results:
            return []

        if not vector_results:
            # BM25-only: pass metadata through
            return [
                {
                    "content": r["content"],
                    "created_at": r.get("created_at", ""),
                    "chunk_idx": r.get("chunk_idx", 0),
                    "total_chunks": 1,
                    "group_id": r.get("group_id", ""),
                    "strength": r.get("strength", 50),
                    "memory_type": r.get("memory_type", "general"),
                    "tags": _parse_tags(r.get("tags", "")),
                    "source": "bm25",
                }
                for r in bm25_results[:n_results]
            ]

        if not bm25_results:
            # Vector-only: fall through to standard search
            return self.search(query, n_results=n_results)

        # -- RRF Fusion --
        fused = rrf_fuse(vector_results, bm25_results, top_k=n_results)

        # Strip the internal rrf_score and source fields from output
        # to match the standard search() format
        clean = []
        for item in fused:
            clean.append({
                "content": item["content"],
                "created_at": item.get("created_at", ""),
                "chunk_idx": item.get("chunk_idx", 0),
                "total_chunks": item.get("total_chunks", 1),
                "group_id": item.get("group_id", ""),
                "strength": item.get("strength", 50),
                "memory_type": item.get("memory_type", "general"),
                "tags": _parse_tags(item.get("tags", "")),
                "rrf_score": item.get("rrf_score", 0.0),
                "source": item.get("source", "hybrid"),
            })
        return clean

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

    @property
    def bm25_ready(self) -> bool:
        """Whether the BM25 index has been built at least once."""
        return self._bm25 is not None and not self._bm25.needs_rebuild

    def recall(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Hybrid search + weighted random sampling for serendipitous recall.

        Uses search_hybrid() to get candidates, then picks `n_results` with
        probability proportional to strength^2. High-strength memories appear
        more often, but low-strength ones still get a look-in — avoids
        deterministic bias where the same top-N results surface every time.

        Returns: same format as search_hybrid().
        """
        candidates = self.search_hybrid(query, n_results=n_results * 4)
        if not candidates:
            return []
        if len(candidates) <= n_results:
            return candidates
        return weighted_sample(candidates, n=n_results, strength_key="strength")

    def delete(
        self,
        query: str = "",
        doc_ids: Optional[List[str]] = None,
        group_id: str = "",
    ) -> int:
        """Delete memories by query (relevance match), specific chunk IDs, or group_id.

        Args:
            query: Delete memories matching this query.
            doc_ids: Specific chunk IDs to delete.
            group_id: Delete all chunks belonging to this group.

        Returns:
            Number of chunks deleted.
        """
        try:
            target_ids: set[str] = set()

            if doc_ids:
                target_ids.update(doc_ids)

            if group_id:
                res = self._collection.get(where={"group_id": group_id})
                if res and res.get("ids"):
                    target_ids.update(res["ids"])

            if query and not target_ids:
                matches = self.search_hybrid(query, n_results=5)
                if matches:
                    top_m = matches[0]
                    gid = top_m.get("group_id", "")
                    if gid:
                        res = self._collection.get(where={"group_id": gid})
                        if res and res.get("ids"):
                            target_ids.update(res["ids"])
                    elif top_m.get("doc_id"):
                        target_ids.add(top_m["doc_id"])

            if not target_ids:
                return 0

            target_list = list(target_ids)
            self._collection.delete(ids=target_list)

            # Update BM25 index & disk cache
            if self._bm25 is not None:
                self._bm25.remove_documents(target_list)
                self._bm25.save_cache(self._bm25_cache_file)

            logger.info("Mihaji: deleted %d chunks (%s)", len(target_list), target_list[:5])
            return len(target_list)
        except Exception as e:
            logger.exception("Mihaji delete failed: %s", e)
            return 0

    def get_stitched_context(
        self,
        group_id: str,
        chunk_idx: int = 0,
        total_chunks: int = 1,
        fallback_content: str = "",
    ) -> str:
        """Fetch surrounding chunks (chunk_idx - 1, chunk_idx, chunk_idx + 1) and stitch them together.

        Args:
            group_id: The UUID group identifier.
            chunk_idx: The current hit chunk index.
            total_chunks: Total chunks in this group.
            fallback_content: Content to return if stitching fails or total_chunks == 1.

        Returns:
            Stitched text seamlessly combining surrounding chunks.
        """
        if total_chunks <= 1 or not group_id:
            return fallback_content

        try:
            start_idx = max(0, chunk_idx - 1)
            end_idx = min(total_chunks - 1, chunk_idx + 1)
            target_ids = [f"{group_id}_{i}" for i in range(start_idx, end_idx + 1)]

            data = self._collection.get(ids=target_ids)
            docs = data.get("documents", []) or []
            metas = data.get("metadatas", []) or []

            if not docs:
                return fallback_content

            pairs = []
            for doc, meta in zip(docs, metas):
                idx = meta.get("chunk_idx", 0) if meta else 0
                pairs.append((idx, doc))
            pairs.sort(key=lambda x: x[0])

            stitched_parts: List[str] = []
            for _, doc in pairs:
                if not stitched_parts:
                    stitched_parts.append(doc)
                else:
                    prev = stitched_parts[-1]
                    overlap_len = CHUNK_OVERLAP
                    merged = False
                    for ol in range(overlap_len + 15, max(0, overlap_len - 20), -1):
                        if ol < len(prev) and ol < len(doc) and prev[-ol:] == doc[:ol]:
                            stitched_parts.append(doc[ol:])
                            merged = True
                            break
                    if not merged:
                        stitched_parts.append(" " + doc)

            return "".join(stitched_parts)
        except Exception as e:
            logger.debug("get_stitched_context failed: %s", e)
            return fallback_content


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_tags(raw: Any) -> List[str]:
    """Parse tags from metadata — stored as comma-separated string."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [t.strip() for t in str(raw).split(",") if t.strip()]


# ------------------------------------------------------------------
# Chunking (from original mem.py)
# ------------------------------------------------------------------

def _chunk_text(text: str, size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for context coherence."""
    if not text:
        return []
    if size <= overlap or size <= 0:
        return [text]
    if len(text) <= size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = size - overlap
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start += step
        if end >= len(text):
            break
    return chunks
