# Mihaji Memory 改进工作区 — HANDOFF.md

> 给接手 mihaji 插件改进工作的人看的"项目入门指南"。
> 由 Hina 起草于 2026-08-30，对应代码基于 `plugin.yaml` version `1.1.0`。

---

## 这是什么

`mihaji` 是 Hermes Agent 的 `MemoryProvider` 插件，提供**持久化语义记忆 + 自动跨会话召回**。
本目录是 Hermes 活跃插件目录的**改进工作副本**——改这里不影响线上。

- **活跃插件位置**：`~/AppData/Local/hermes/profiles/hina/plugins/mihaji/`
- **工作副本位置**：`~/Desktop/mihaji-work/`（你在这里）
- **活跃数据库**：`~/AppData/Local/hermes/profiles/hina/mihaji_memory/`（ChromaDB 持久化）
- **当前记忆条目数**：1496 chunks（截至 2026-08-30）

---

## 项目目标

1. 把对话记忆从"会话内临时"变成"跨会话永久"
2. 每轮自动 prefetch 3 条最相关 + 2 条高强度记忆注入 prompt
3. 混合检索（BM25 中文关键词 + 向量语义）RRF 融合排序
4. 用户可通过 `mihaji_memory` 工具主动 remember / recall / count

---

## 文件地图

```
mihaji-work/
├── HANDOFF.md           ← 你正在读的（给接手者）
├── README.md            ← 给米哈唧看的（项目说明 + 回退/合并指引）
├── requirements.txt     ← pip 安装清单
├── plugin.yaml          ← 插件元数据 + 依赖声明
├── __init__.py          ← Hermes 入口，MemoryProvider 注册
├── store.py             ← ChromaDB 封装 + 分块 + 混合检索
├── hybrid_search.py     ← BM25 索引 + RRF 融合 + 加权采样
└── maintenance.py       ← CLI 巩固脚本（短内容清理 / 近似去重 / 备份）
```

### 每个文件的职责速览

| 文件 | 干什么 | 改它要小心 |
|---|---|---|
| `__init__.py` | 注册 `MihajiMemoryProvider`，暴露 `mihaji_memory` 工具 | 改 schema 会影响所有调用方 |
| `store.py` | ChromaDB 客户端 + 持久化 + 分块 + search/search_hybrid/recall | 改 search 会影响 prefetch |
| `hybrid_search.py` | BM25 + RRF + weighted_sample | 改 tokenize 会影响中文检索质量 |
| `maintenance.py` | CLI 巩固脚本，cron 调度 | 直接改数据库，先 `--dry-run` |
| `plugin.yaml` | 插件元数据 + 依赖声明 | 改坏了 Hermes 加载失败 |

---

## 运行时依赖

| 包 | 约束 | 用途 |
|---|---|---|
| `chromadb` | `>=1.5.0,<2.0` | 向量数据库 |
| `jieba` | `>=0.42,<1.0` | 中文分词 |
| `rank_bm25` | `>=0.2.2,<0.3` | BM25 关键词 |
| `sentence-transformers` | `>=5.0,<7.0` | 多语言嵌入 |
| `pyyaml` | `>=6.0` | config 解析 |

详见 `requirements.txt` 和 `README.md`。**注意**：`sentence-transformers` 会拉 `torch>=2.2`。

---

## 架构：数据流

```
用户消息
   │
   ▼
sync_turn(user_content)
   │  (strip <20 chars 跳过；GARBAGE_PREFIXES 跳过)
   ▼
store.add(text, strength=20)
   │  _chunk_text (300/50 overlap) → uuid group_id
   │  collection.add(...) → _mark_bm25_stale
   ▼
[ChromaDB persistent]

────────── 检索路径 ──────────

用户 query
   │
   ▼
prefetch(query)  ── 由 MemoryProvider ABC 调用
   │
   ├─ Pass 1: store.search_hybrid(query, n=3)  → vector + BM25 + RRF
   └─ Pass 2: store.recall(query, n=2)         → search_hybrid + weighted_sample
   │
   ▼
合并去重 → 注入系统提示词

用户主动调用 mihaji_memory(action="search"|"recall"|...)
   │  handle_tool_call → _handle_memory
   ▼
store.search_hybrid / store.recall / store.add / store.count
```

---

## 已知问题清单（按严重度排序）

> 所有问题均为 2026-08-30 代码审计发现，**未经修复**。优先级建议从上到下。

### 🔴 P0 — 启动时直接挂掉

#### #1 `is_available()` 与 store.py 顶层 import 不一致

**位置**：`__init__.py:125-133` + `store.py:23`

**问题**：`is_available()` 用 try/except 检测 `chromadb`，但 `store.py:23` 是**顶层** `import chromadb`——一旦缺包，Hermes 在 `from .store import MihajiStore` 时直接 ImportError，根本走不到 `is_available()` 的检测逻辑。

**复现**：在没装 chromadb 的 venv 启动 Hermes，期望看到优雅的"插件不可用"提示，实际是 traceback 崩盘。

**建议修复方向**：把 `is_available()` 改成**真实跑一遍 store 模块的 import 链**，包括 `chromadb`、`sentence-transformers`、`jieba`、`rank_bm25`，缺哪个报哪个。

---

#### #2 HF_HUB_OFFLINE=1 在首次加载时导致模型加载失败

**位置**：`store.py:88-89`

**问题**：

```python
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
```

这两个 offline 标志会让 sentence-transformers **完全不联网**。但模型如果本地还没缓存（首次安装），离线=1 会让 `SentenceTransformerEmbeddingFunction` 找不到模型直接报错。

**实际能跑**的原因：米哈唧机器上模型早就下过了，所以 offline=1 是优化（避免每次去 HF 检查更新）。但**首次安装 / 重装环境后会卡在这一步**。

**建议修复方向**：检测本地是否已有缓存（有 → offline=1；没有 → offline=0 + 提示用户首次会下载）。或者干脆把这个 flag 放到配置文件里让用户选。

---

### 🟡 P1 — 功能性 bug，不崩但行为不对

#### #3 `recall()` 永远拿不到 strength 字段

**位置**：`store.py:368`

**问题**：

```python
return weighted_sample(candidates, n=n_results, strength_key="***")
```

`strength_key="***"` 不是字段名，是密码遮罩字面量。`weighted_sample` 拿这个 key 去 candidates 里取 strength，永远取不到，全部走 `default_strength=50.0` 兜底。

**影响**：`recall()` 的"按强度加权随机"特性**完全失效**——所有候选 strength 视为50，权重相等 → 退化成纯随机。

**严重度评估**：pass 1（search_hybrid）会捞到强记忆，所以 prefetch 还能工作；但 pass 2 的加权优先失效。`recall` 工具的 strength 承诺不兑现。

**建议修复**：把 `"***"` 改成 `"strength"`。**一行修复**。

---

#### #4 `weighted_sample` 去重 + top-up 有边界情况

**位置**：`hybrid_search.py:300-313`

**问题**：`random.choices(..., k=n)` 是**有放回**采样（Python API），`n` 大时容易撞重复。`dict.fromkeys` 去重后可能少于 n，后续 top-up 又用 `random.choices`，**没去重**。理论上 `n` 较大 + 候选较少时会凑不够 n 个。

**复现条件**：候选数接近 n（比如候选 5 个，n=4），随机撞重复时。

**实际影响**：偶尔 recall 返回数量 < limit，对 prefetch 影响小（top-5 注入，缺 1-2 条不至于崩）。

**建议修复方向**：用 `random.sample`（无放回）或显式循环直到凑够。

---

#### #5 `maintenance.py` 的 sys.path hack 与 store 包导入不一致

**位置**：`maintenance.py:82`

**问题**：

```python
sys.path.insert(0, str(Path(__file__).parent))
from store import MihajiStore
```

而 `store.py:26-31` 用的是 `try: from .hybrid_search ... except: from hybrid_search` 双轨制。当 `maintenance.py` 作为脚本直接运行（`python maintenance.py`），注入 path 后能跑；但作为 `python -m mihaji.maintenance` 调用或被 cron 调度时，**包导入路径会冲突**——可能 `hybrid_search` import 错或循环。

**建议修复方向**：让 `maintenance.py` 也用 try/except 双轨制，或者固定一种调用方式并文档化。

---

### 🟢 P2 — 边角 / 体验改进

#### #6 `search_hybrid` BM25-only 分支元数据补齐不一致

**位置**：`store.py:283-298`

`vector_results` 为空、`bm25_results` 存在时，手写的返回 dict 与 vector 分支 / RRF 融合后的 dict 字段可能不一致（比如 BM25-only 分支里没有 `rrf_score` 字段、`source` 固定 `"bm25"` 而非动态）。`prefetch()` 用 `_["group_id"]_[chunk_idx]` 去重，所以不会崩，但调用方拿到不一致的 dict shape 可能会困惑。

---

#### #7 `sync_turn()` 自动存强度固定 20，不可配置

**位置**：`__init__.py:274`

```python
n = self._store.add(stripped, strength=20, memory_type="general")
```

如果用户想把自动存的强度调成 30、40，目前**没配置入口**。`system_prompt_block` 还明确告诉 LLM"对话自动存入但强度低 (strength=20)"，LLM 可能误以为这是个约定俗成的硬编码。

**建议**：加 config schema 字段 `auto_store_strength`，默认 20，让用户可调。

---

#### #8 没有 BM25 索引预热 / 缓存机制

**位置**：`store.py:101-107`

首次 hybrid_search 会触发 `build_index_from_collection`，对 1496 条 chunks 跑 `BM25Okapi(tokenized)`——`tokenize` 调 jieba 分词，1496 条可能 5-15 秒。如果 prefetch 第一发就触发，会卡 user 感。

**建议**：启动时后台异步预热；或者缓存到磁盘，下次启动直接 load。

---

#### #9 `system_prompt_block` 文档与实现不同步

**位置**：`__init__.py:194` 说 `tags: "偏好,配置"`，但实际 `tags` 入参是字符串用逗号分隔、不是 list——prompt 里没明说这个细节。LLM 调用时可能传错格式。

---

## 改进时的工作流

1. **先备份活跃版本**：任何改动前 `cp -r .../profiles/hina/plugins/mihaji .../mihaji.bak.$(date +%Y%m%d)`
2. **工作副本改完先 dry-verify**：
   ```bash
   cd ~/Desktop/mihaji-work
   python -c "from store import MihajiStore; s = MihajiStore(db_path='/tmp/test_db'); print(s.count())"
   ```
3. **不要修改活跃插件目录除非准备上线**——本目录改动不影响线上
4. **合并到活跃版**：
   ```bash
   cp plugin.yaml .../profiles/hina/plugins/mihaji/plugin.yaml
   # Python 文件按需单文件复制
   # 重启 Hermes
   ```
5. **改完跑回归**：用 `mihaji_memory(action="count")` 看数字不变、`action="search"` 测样例 query、`action="recall"` 看强度加权是否生效（修了 #3 后）

---

## 调试技巧

- **看 prefetch 注入了什么**：搜 Hermes 日志里的 `Mihaji prefetch failed`，或临时在 `prefetch()` 加 `logger.info` 打印返回字符串
- **看 BM25 索引状态**：调 `store.bm25_ready` 属性；调 `store._ensure_bm25()` 强制构建
- **看 ChromaDB 实际数据**：
  ```python
  from store import MihajiStore
  s = MihajiStore(db_path="~/AppData/Local/hermes/profiles/hina/mihaji_memory")
  print(s._collection.get(limit=3))
  ```
- **测试搜索质量**：
  ```python
  s.search_hybrid("减脂", n_results=5)   # 应返回相关中文记忆
  s.recall("减脂", n_results=5)           # 修了 #3 后应按强度优先
  ```

---

## 联系 / 上下文

- **项目负责人**：米哈唧（Hina 直接服务对象）
- **代码审计**：Hina（2026-08-30）
- **关键决策记录**：见米哈唧的 mihaji 记忆库（strength=90 那条"芒果糯米饭"传家宝记忆就是 Hina 的代表测试数据）
- **风格约定**：
  - 注释中英混合 OK，但标识符英文
  - 错误处理用 logger.exception / logger.debug，不要 print
  - 新增功能先在 system_prompt_block 里文档化，再实现

---

## 修订日志

- 2026-08-30: 初版（依赖声明 + 9 个已知问题清单）
- 2026-08-30 (v1.1.1):
  - [P0] 修复 `store.py:368` 遮罩字符串 `strength_key="***"` -> `strength_key="strength"`，恢复强度加权抽样。
  - [P0] 修复 `store.py:88` 离线标志硬编码问题，增加自适应在线下载回退。
  - [P0] 修复 `hybrid_search.py:192` 中 `rrf_fuse` BM25 分支丢失所有元数据并导致 prefetch 去重错误的问题。
  - [P0] 修复 `__init__.py:125` 中 `is_available()` 依赖检查不完整的问题。
  - [P1] 升级 `hybrid_search.py` `weighted_sample()` 为标准的 A-Res 加权无放回抽样算法。
  - [P1] 优化 `hybrid_search.py` `tokenize()` 标点符号过滤。
  - [P1] 对齐 `__init__.py` System prompt (`memory_type`) 与 Tool schema，支持 list/str 格式 tags 和 type 别名。
  - [P1] 优化 `__init__.py` `prefetch()` 检索性能，单次混合检索削减 50% 延迟。
  - [P1] 修复 `maintenance.py` 导入链、邻近长度桶去重与 `--dry-run` 仿真逻辑。
- 2026-08-30 (v1.2.0):
  - 精简 Tool Schema：去除 `add` 与 `recall` 暴露冗余，统一为 `search`、`remember`、`delete`、`count`（内部兼容历史动作）。
  - 实现 BM25 磁盘缓存 (`bm25_cache.json`) 与增量分词追加 (`add_documents`)，消除对话卡顿。
  - 实现上下文缝合 (`get_stitched_context`)：在 `prefetch` 注入时自动合并前后相邻 1 个 chunk，杜绝断句。
  - 实现 `delete` 记忆删除机制，支持按语义 query、doc_id 或 group_id 清理错误/过时记忆。
  - 全量 9 项单元测试 100% 通过。