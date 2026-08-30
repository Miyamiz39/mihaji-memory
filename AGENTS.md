# Mihaji Memory — Agent 维护与交接指南 (AGENTS.md)

> **给接手 mihaji-memory 维护与演进的 AI Agent / 开发者**。
> 本文件记录了项目的架构设计、核心机制、已解决的历史缺陷、测试流程及后续演进方向。

---

## 📌 项目概述

`mihaji-memory` 是 [Hermes Agent](https://hermes-agent.nousresearch.com) 的原生 `MemoryProvider` 插件，提供**持久化多语言语义记忆 + 关键词双轨混合检索 + 自动跨会话召回**。

- **GitHub 仓库**：[https://github.com/Miyamiz39/mihaji-memory](https://github.com/Miyamiz39/mihaji-memory)
- **开源协议**：MIT License
- **插件版本**：`1.2.0`
- **活跃数据库位置**：`~/.hermes/profiles/<profile>/mihaji_memory/` (Windows: `%LOCALAPPDATA%\hermes\profiles\<profile>\mihaji_memory\`)
- **活跃插件安装位置**：`~/.hermes/profiles/<profile>/plugins/mihaji/` (Windows: `%LOCALAPPDATA%\hermes\profiles\<profile>\plugins\mihaji\`)

---

## 🗂️ 仓库目录结构

本仓库采用标准的独立插件仓库组织结构：

```
mihaji-memory/
├── README.md               # 中文公开说明文档
├── README.en.md            # English Documentation
├── requirements.txt        # pip 依赖清单
├── LICENSE                 # MIT 开源协议
├── AGENTS.md               # 你正在读的（Agent 维护与交接指南）
├── .gitignore              # Git 忽略规则
└── mihaji/                 # 核心插件目录（软链接或复制至 Hermes plugins/）
    ├── __init__.py         # Hermes 入口，MemoryProvider 注册与 Tool Schema
    ├── store.py            # ChromaDB 封装、分块、上下文缝合、增量更新与删除
    ├── hybrid_search.py    # BM25 增量索引、磁盘缓存、RRF 融合与 A-Res 加权采样
    ├── maintenance.py      # CLI 离线巩固脚本（短内容过滤 / Jaccard 近似去重 / 备份）
    └── plugin.yaml         # Hermes 插件元数据与依赖声明
```

### 核心模块职责

| 模块文件 | 主要职责 | 修改注意事项 |
|---|---|---|
| `mihaji/__init__.py` | 注册 `MihajiMemoryProvider`，声明 `mihaji_memory` Tool Schema，实现 `prefetch()` 与 `sync_turn()` | 保持 Tool Schema 精简，改动需同步修改 `system_prompt_block()` |
| `mihaji/store.py` | ChromaDB 客户端封装，300字切块与50字重叠，`get_stitched_context()` 上下文缝合，增量更新与删除 | 涉及检索与注入效果，修改需跑全量单元测试 |
| `mihaji/hybrid_search.py` | BM25 索引构建、`bm25_cache.json` 磁盘缓存、RRF 融合排序、A-Res 加权无放回采样 | 分词与标点过滤直接影响中文关键词检索召回率 |
| `mihaji/maintenance.py` | 独立 CLI 巩固工具，支持 `--dry-run` 仿真、Jaccard 跨桶去重与 JSON 导出 | 直接修改数据库，执行前务必备份 |
| `mihaji/plugin.yaml` | Hermes 插件元数据声明 | 版本号与 hook 声明必须与实际实现匹配 |

---

## 🏗️ 架构与数据流

```
【用户发送消息】
   │
   ▼
prefetch(query) ── Hermes MemoryProvider 机制触发
   │
   ├─ Pass 1: store.search_hybrid(query, n=10) ── Vector + Jieba BM25 + RRF 融合重排
   ├─ Pass 2: 取 Top-3 精准命中 + 剩余池执行 A-Res 加权采样抽 2 条 (strength² 权重放大)
   └─ Pass 3: store.get_stitched_context() ── 自动缝合命中块相邻前后 chunk (chunk_idx ± 1)
   │
   ▼
合并去重段落 → 自动注入系统提示词 (System Prompt)

【LLM 思考与回复】
   │
   ▼
sync_turn(user_content) ── 自动记忆钩子
   │ (过滤 <20 字符噪音与垃圾前缀)
   ▼
store.add(text, strength=20, memory_type="general")
   │
   ├─ _chunk_text (300字/50字重叠) → UUID group_id
   ├─ collection.add(...) 稠密向量入库
   └─ bm25.add_documents(...) 增量追加分词并持久化 bm25_cache.json (零主线程卡顿)
   ▼
[ChromaDB 持久化存储]

【用户/Agent 主动调用】
mihaji_memory(action="search" | "remember" | "delete" | "count")
   │  handle_tool_call → _handle_memory
   ▼
store.search_hybrid / store.add / store.delete / store.count
```

---

## 📜 历史问题解决记录 (Changelog v1.1.0 ~ v1.2.0)

所有早期代码审计中发现的 P0/P1/P2 缺陷均已全部彻底解决：

1. **[已修复 P0] 强度加权采样失效**：修复 `store.py` 中 `strength_key="***"` 脱敏字符串导致的权重降级，恢复真实的记忆强度优先加权。
2. **[已修复 P0] 离线模型加载卡死**：增加自适应回退机制，首次安装无本地缓存时自动拉取模型，已有缓存时启用离线加速。
3. **[已修复 P0] RRF 融合元数据丢失**：修复 `hybrid_search.py` 中 BM25 分支丢失 `group_id` / `chunk_idx` 导致 `prefetch` 去重误伤的缺陷。
4. **[已修复 P0] 依赖检测链不完整**：`is_available()` 校验完整的 import 链（`chromadb`, `sentence_transformers`, `jieba`, `rank_bm25`）。
5. **[已优化 P1] A-Res 加权无放回抽样**：替换旧有带放回抽样为标准 Efraimidis-Spirakis A-Res 加权采样算法。
6. **[已优化 P1] BM25 磁盘缓存与增量追加**：实现 `bm25_cache.json` 磁盘持久化与 `add_documents` 增量追加，启动与交互达毫秒级响应。
7. **[已优化 P1] 上下文自动缝合 (Chunk Stitching)**：`get_stitched_context` 自动合并前后相邻 1 个分块，彻底消除断句断意。
8. **[已优化 P1] 工具体系精简**：收敛为 `search`, `remember`, `delete`, `count` 4 大动作，底层保持旧动作（`add`/`recall`/`forget`）静默兼容。
9. **[已优化 P1] 离线巩固脚本规范化**：修复 `maintenance.py` 导入链与跨桶 Jaccard 去重计算。
10. **[已重构 P0] 直接托管本地嵌入推理架构**：绕过 ChromaDB 内部的黑盒网络调用与 repo ID 依赖，在 Python 层直接利用本地快照绝对路径加载 `SentenceTransformer` 并显式传递 `embeddings` / `query_embeddings`，100% 根除境外 HuggingFace HEAD 探测导致的 `getaddrinfo failed` 与 `Client closed` 异常。
11. **[已优化 P0] 异步预热与冷启动 BM25 极速降级**：插件初始化时在后台守护线程异步加载模型，首轮对话若模型未就绪则 0 毫秒阻塞降级为 BM25 纯关键词检索，彻底解决 Hermes 网关 8.0s 超时截断问题。
12. **[已加固 P1] 显式禁用 ChromaDB 遥测**：注入 `anonymized_telemetry=False`，彻底免疫第三方 OpenTelemetry / Protobuf 升级引起的底层符号冲突。

---

## 🧪 自动化测试与验证

仓库附带了全流程单元测试，覆盖分词、加权采样、RRF 融合、BM25 增量缓存、上下文缝合、CRUD 生命周期与维护脚本。

### 运行测试

在配置好依赖的 Python 环境（推荐 Hermes venv）中运行：

```bash
# 执行测试套件
python -m unittest discover -s . -p "*test*.py"
```

---

## 🔄 开发与上线工作流

无论本项目移动到哪个本地目录，标准上线与同步流程如下：

1. **在工作目录修改与测试**：
   - 保持 opinionated 极简哲学，避免引入冗余的可调配置项。
2. **同步至 Hermes 活跃插件目录**：
   ```bash
   # Linux / macOS:
   rsync -av --exclude '__pycache__' ./mihaji/ ~/.hermes/profiles/<profile>/plugins/mihaji/

   # Windows (PowerShell):
   Copy-Item -Path ".\mihaji\*" -Destination "$env:LOCALAPPDATA\hermes\profiles\<profile>\plugins\mihaji" -Recurse -Force
   ```
3. **Git 同步与发布**：
   ```bash
   git add .
   git commit -m "feat/fix: <description>"
   git push origin main
   ```

---

## 🚀 未来演进方向 (Roadmap)

- [ ] **时间遗忘曲线 (Temporal Decay)**：基于艾宾浩斯记忆模型，随时间动态衰减未访问的非核心记忆强度。
- [ ] **记忆主题聚类 (Thematic Clustering)**：按标签或语义中心进行自动聚类，支持超长记忆库的分层结构化摘要。
- [ ] **跨设备记忆同步**：轻量级导出与端对端同步机制。
