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
        "语义记忆库 — 用向量搜索找回过去的对话和约定。\n"
        "ACTIONS:\n"
        "• search — 根据关键词或语义搜索过往记忆，返回相关片段。\n"
        "• add — 手动添加一段记忆（对话自动记录，这个用于补充细节）。\n"
        "• count — 查看记忆库总条目数。\n"
        "\n"
        "重要: 每次回复前先用 search 查一下有没有相关记忆！"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "add", "count"],
            },
            "query": {
                "type": "string",
                "description": "搜索关键词或语义查询（action='search' 时必填）",
            },
            "text": {
                "type": "string",
                "description": "要手动添加的记忆内容（action='add' 时必填）",
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数量（默认 5）",
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
        """Check that chromadb and sentence-transformers are importable."""
        try:
            import chromadb  # noqa: F401
            from chromadb.utils import embedding_functions  # noqa: F401

            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Set up ChromaDB with persistent storage under HERMES_HOME."""
        from .store import MihajiStore

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
                "记忆库是空的哦。用 mihaji_memory(action='add') 来存重要的事，"
                "每次回复前用 mihaji_memory(action='search') 查查有没有相关记忆。"
            )
        return (
            f"# Mihaji 记忆库 🐾\n"
            f"已存 {count} 条记忆片段。用 mihaji_memory(action='search') "
            f"搜索过往，用 mihaji_memory(action='add') 手动添加。"
        )

    # -- Prefetch -------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Semantic recall before each turn. Returns formatted memories or ''."""
        if not self._store or not query:
            return ""
        try:
            results = self._store.search(query, n_results=5)
            if not results:
                return ""

            lines = ["## 相关回忆 🐾"]
            for r in results:
                lines.append(
                    f"- [{r['created_at']}] {r['content']}"
                )
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
            n = self._store.add(stripped)
            logger.debug("Mihaji: auto-stored %d chunks", n)
        except Exception as e:
            logger.debug("Mihaji sync_turn failed: %s", e)

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
                results = self._store.search(query, n_results=limit)
                if not results:
                    return json.dumps(
                        {"results": [], "count": 0, "hint": "没找到相关记忆"},
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {"results": results, "count": len(results)},
                    ensure_ascii=False,
                )

            elif action == "add":
                text = args.get("text", "")
                if not text:
                    return tool_error("add 需要 'text' 参数")
                n = self._store.add(text)
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
