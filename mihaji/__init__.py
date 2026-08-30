"""
Mihaji Memory — ChromaDB-backed MemoryProvider plugin for Hermes Agent.

Transforms the original mihaji-memory Skill into a native MemoryProvider,
giving the agent automatic cross-session semantic recall without manual
/skill loading.

Config (in ~/.hermes/config.yaml):
  plugins:
    mihaji:
      db_path: ~/.hermes/mihaji_memory
      model_name: paraphrase-multilingual-MiniLM-L12-v2
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

MIHAJI_MEMORY_SCHEMA = {
    "name": "mihaji_memory",
    "description": (
        "语义记忆库 — 混合检索（语义+关键词）找回过去的对话和约定，或存储长期偏好。\n"
        "ACTIONS:\n"
        "• search — 根据关键词或语义混合搜索过往记忆，返回最相关片段。\n"
        "• remember — 主动存储一条重要记忆，可指定强度、类型和标签。\n"
        "• delete — 删除记忆（指定查询关键词或条目 ID 进行清理）。\n"
        "• count — 查看记忆库总条目数。\n"
        "\n"
        "重要: 每次回复前系统会自动注入相关回忆，必要时可用 search 补充查询！"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "remember", "delete", "count"],
            },
            "query": {
                "type": "string",
                "description": "搜索关键词或待删除的记忆内容（action='search' 或 'delete' 时使用）",
            },
            "text": {
                "type": "string",
                "description": "记忆内容（action='remember' 时必填）",
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数量（search 默认 5）",
            },
            "strength": {
                "type": "integer",
                "description": "记忆强度 1-100，越高越重要（action='remember' 可选，默认 70）",
            },
            "memory_type": {
                "type": "string",
                "description": "记忆类型：knowledge/preference/event/task/general（action='remember' 可选，默认 general）",
            },
            "tags": {
                "type": "string",
                "description": "逗号分隔的标签文本或列表（action='remember' 可选）",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_plugin_config(hermes_home: str) -> dict:
    """Load the mihaji section from config.yaml."""
    from hermes_cli.config import cfg_get

    config_path = Path(hermes_home) / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "mihaji", default={}) or {}
    except Exception:
        return {}


def _default_db_path(hermes_home: str) -> str:
    return str(Path(hermes_home) / "mihaji_memory")


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------


class MihajiMemoryProvider(MemoryProvider):
    """ChromaDB vector memory with Chinese-friendly multilingual embeddings."""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._store = None
        self._session_id = ""

    # -- Required properties / methods ----------------------------------------

    @property
    def name(self) -> str:
        return "mihaji"

    def is_available(self) -> bool:
        """Check that chromadb, sentence-transformers, jieba, and rank_bm25 are importable."""
        try:
            import chromadb  # noqa: F401
            from chromadb.utils import embedding_functions  # noqa: F401
            import jieba  # noqa: F401
            import rank_bm25  # noqa: F401
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Set up ChromaDB with persistent storage under HERMES_HOME."""
        try:
            from .store import MihajiStore
        except ImportError:  # Hermes loads user plugins as top-level module; fall back to absolute import
            from store import MihajiStore

        hermes_home = kwargs.get("hermes_home", "~/.hermes")
        hermes_home = str(Path(hermes_home).expanduser())

        db_path = self._config.get("db_path") or _default_db_path(hermes_home)
        model_name = self._config.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")

        # Expand ~ and $HERMES_HOME in db_path
        db_path = Path(db_path).expanduser()
        db_path = str(db_path).replace("$HERMES_HOME", hermes_home)

        self._store = MihajiStore(db_path=db_path, model_name=model_name)
        self._session_id = session_id

        logger.info(
            "Mihaji: initialized (db=%s, model=%s, chunks=%d)",
            db_path,
            model_name,
            self._store.count(),
        )

    # -- System prompt --------------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        count = self._store.count()
        if count == 0:
            return (
                "# Mihaji 记忆库 🐾\n"
                "记忆库还是空的哦。可以用 mihaji_memory 来操作：\n"
                "  • search — 搜索相关过往记忆\n"
                "  • remember — 学到重要事实、偏好时主动存下来\n"
                "  • delete — 删除或遗忘错误/过时的记忆\n"
                "  • count — 查看记忆库条目数"
            )
        return (
            f"# Mihaji 记忆库 🐾\n"
            f"已存 {count} 条记忆片段。\n\n"
            f"### 操作说明\n"
            f"`mihaji_memory(action='search', query='...')` — 搜索相关记忆。\n"
            f"`mihaji_memory(action='remember', text='...', strength=70, memory_type='knowledge', tags='标签1,标签2')` — 记录重要信息。\n"
            f"`mihaji_memory(action='delete', query='...')` — 删除或遗忘指定记忆。\n"
            f"`mihaji_memory(action='count')` — 查看记忆库条目数。\n\n"
            f"### 主动记忆场景\n"
            f"**什么时候该 remember？**\n"
            f"  • 用户说出了重要的个人信息（职业、学校、城市、联系方式）\n"
            f"  • 用户表达了明确的偏好（喜欢什么、讨厌什么、习惯什么）\n"
            f"  • 用户纠正了你（别忘了这个，下次要记住）\n"
            f"  • 你和用户共同做了决定（选定了某个方案、工具、方向）\n"
            f"  • 用户交代了需要长期记住的约定或规则\n\n"
            f"**记忆属性：**\n"
            f"  • strength: 1-100，越高越重要（默认70，极重要的给90+）\n"
            f"  • memory_type: knowledge / preference / event / task / general\n"
            f"  • tags: 逗号分隔的关键词标签（或列表）\n\n"
            f"**自动机制：**\n"
            f"  • 每轮对话前会自动根据当前话题召回最相关的记忆片段注入提示词，并缝合前后上下文保证完整语义。"
        )

    # -- Prefetch -------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Two-pass prefetch: relevance top-3 + strength-weighted top-2.

        Pass 1: search_hybrid — 一次性检索候选池（含向量与 BM25 RRF 融合）
        Pass 2: 前 3 条精准命中 + 剩余池中按 strength² 加权抽 2 条
        合并去重 + 前后 chunk 上下文缝合 → top-5 注入提示词
        """
        if not self._store or not query:
            return ""
        try:
            candidates = self._store.search_hybrid(query, n_results=10)
            if not candidates:
                return ""

            top_relevant = candidates[:3]
            remaining_pool = candidates[3:]

            if remaining_pool:
                try:
                    from .hybrid_search import weighted_sample
                except ImportError:
                    from hybrid_search import weighted_sample
                top_weighted = weighted_sample(remaining_pool, n=2, strength_key="strength")
            else:
                top_weighted = []

            # Merge: key by group_id+chunk_idx, keep order (relevance first)
            seen: set = set()
            merged: list = []
            for item in top_relevant + top_weighted:
                gid = item.get("group_id", "")
                cidx = item.get("chunk_idx", 0)
                key = f"{gid}_{cidx}" if gid else item.get("content", "")
                if key not in seen:
                    seen.add(key)
                    merged.append(item)

            if not merged:
                return ""

            lines = ["## 相关回忆 🐾"]
            for r in merged[:5]:
                created_at = r.get("created_at", "")
                # Seamlessly stitch surrounding chunk_idx - 1 and chunk_idx + 1 if available
                content = self._store.get_stitched_context(
                    group_id=r.get("group_id", ""),
                    chunk_idx=r.get("chunk_idx", 0),
                    total_chunks=r.get("total_chunks", 1),
                    fallback_content=r.get("content", ""),
                )
                if created_at:
                    lines.append(f"- [{created_at}] {content}")
                else:
                    lines.append(f"- {content}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug("Mihaji prefetch failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No background prefetch needed — search is fast (local)."""
        pass

    # -- Sync turn (auto-memorize) --------------------------------------------

    # Background review prompts patterns — Hermes spawns silent agent forks
    # every N turns to review the conversation. These system prompts get fed
    # as user_content and MUST NOT be auto-stored into mihaji-memory.
    _GARBAGE_PREFIXES = (
        "Review the conversation above and consider saving to memory",
        "Review the conversation above and update two things",
        "Review the conversation above and update the skill library",
    )

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Auto-store user messages as memories after each turn."""
        if not self._store or not user_content:
            return
        # Only store messages of substance (>20 chars after stripping)
        stripped = user_content.strip()
        if len(stripped) < 20:
            return
        # Skip background review prompts — they're system-level noise, not real chat
        if any(stripped.startswith(p) for p in self._GARBAGE_PREFIXES):
            logger.debug("Mihaji: skipped background review prompt")
            return
        try:
            n = self._store.add(stripped, strength=20, memory_type="general")
            logger.debug("Mihaji: auto-stored %d chunks (strength=20)", n)
        except Exception as e:
            logger.debug("Mihaji sync_turn failed: %s", e)

    def on_session_end(self, session_id: str = "", **kwargs) -> None:
        """Hook called when a session ends."""
        logger.debug("Mihaji: session %s ended", session_id)

    # -- Tools ----------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [MIHAJI_MEMORY_SCHEMA]

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        if tool_name != "mihaji_memory":
            return tool_error(f"Unknown tool: {tool_name}")
        return self._handle_memory(args)

    def _handle_memory(self, args: dict) -> str:
        try:
            action = args["action"]

            if action == "search":
                query = args.get("query", "")
                if not query:
                    return tool_error("search 需要 'query' 参数")
                limit = int(args.get("limit", 5))
                results = self._store.search_hybrid(query, n_results=limit)
                if not results:
                    return json.dumps(
                        {"results": [], "count": 0, "hint": "没找到相关记忆"},
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {"results": results, "count": len(results)},
                    ensure_ascii=False,
                )

            elif action == "recall":
                query = args.get("query", "")
                if not query:
                    return tool_error("recall 需要 'query' 参数")
                limit = int(args.get("limit", 5))
                results = self._store.recall(query, n_results=limit)
                if not results:
                    return json.dumps(
                        {"results": [], "count": 0, "hint": "没找到相关记忆"},
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {"results": results, "count": len(results)},
                    ensure_ascii=False,
                )

            elif action in ("delete", "forget"):
                query = args.get("query", "")
                doc_id = args.get("doc_id", "")
                group_id = args.get("group_id", "")
                if not query and not doc_id and not group_id:
                    return tool_error("delete 需要 'query'、'doc_id' 或 'group_id' 参数")
                doc_ids = [doc_id] if doc_id else None
                deleted = self._store.delete(query=query, doc_ids=doc_ids, group_id=group_id)
                return json.dumps(
                    {"status": "deleted", "deleted_chunks": deleted, "query": query},
                    ensure_ascii=False,
                )

            elif action == "remember":
                text = args.get("text", "")
                if not text:
                    return tool_error("remember 需要 'text' 参数")
                strength = int(args.get("strength", 70))
                memory_type = args.get("memory_type") or args.get("type", "general")
                tags_raw = args.get("tags", "")
                if isinstance(tags_raw, list):
                    tags = [str(t).strip() for t in tags_raw if str(t).strip()]
                elif isinstance(tags_raw, str):
                    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
                else:
                    tags = []
                # Clamp strength
                strength = max(1, min(100, strength))
                n = self._store.add(text, strength=strength, memory_type=memory_type, tags=tags)
                return json.dumps(
                    {
                        "status": "remembered",
                        "chunks": n,
                        "strength": strength,
                        "type": memory_type,
                        "memory_type": memory_type,
                        "tags": tags,
                    },
                    ensure_ascii=False,
                )

            elif action == "add":
                # Backward-compatibility alias -> remember
                text = args.get("text", "")
                if not text:
                    return tool_error("add 需要 'text' 参数")
                n = self._store.add(text, strength=50, memory_type="general")
                return json.dumps(
                    {"status": "added", "chunks": n},
                    ensure_ascii=False,
                )

            elif action == "count":
                return json.dumps(
                    {"count": self._store.count()},
                    ensure_ascii=False,
                )

            else:
                return tool_error(f"未知 action: {action}")

        except KeyError as exc:
            return tool_error(f"缺少参数: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Shutdown -------------------------------------------------------------

    def shutdown(self) -> None:
        self._store = None

    # -- Config ---------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "db_path",
                "description": "记忆数据库存储路径（留空则自动使用 profile 下的默认路径）",
                "default": "",
            },
            {
                "key": "model_name",
                "description": "sentence-transformers 模型名或本地路径",
                "default": "paraphrase-multilingual-MiniLM-L12-v2",
            },
        ]
        
    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to config.yaml under plugins.mihaji."""
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml

            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["mihaji"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.warning("Mihaji save_config failed: %s", e)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register mihaji with the memory plugin system."""
    hermes_home = getattr(ctx, "hermes_home", "~/.hermes")
    config = _load_plugin_config(hermes_home)
    provider = MihajiMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
