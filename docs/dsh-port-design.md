# Mihaji Memory — DSH 移植设计 (Design / 2026-09)

把 `mihaji-memory` 从 **Hermes 原生 MemoryProvider** 移植为 **DeepSeek Harness (DSH) profile bundle 插件**。
目标：让 DSH 的 agent 拥有 mihaji 的「多语言混合记忆 + 每步自动召回」。
**部署目标 profile = `qqbot`（hina QQ 网关，PM2 app `hina`）**；web profile 只作开发验证台，最终不含 mihaji
（2026-09 迁移定案：用户「只是希望把 mihaji 放在 qqbot profile 里」）。可选补一个
`session_search`（搜 DSH 自己的会话历史）。

## 0. 背景澄清

- Hermes 时代，mihaji（ChromaDB 记忆）与 `session_search_tool.py`（读 Hermes `state.db` 会话）是**两套独立数据源**。
- 本次 DSH 移植的**数据源取向**（已与用户确认）：
  - 记忆（mihaji 本体）→ **DSH 原生 JSON 库 + 进程内本地 q8 嵌入**（已落地，见 §5）。
  - `session_search`（如做）→ 搜 **DSH 自己的会话历史**（`ctx.sessionQuery`，FTS5 后端已挂载），
    不是去读 Hermes 的 state.db。
- 用户取舍：**长期记忆为主**，先定架构后议存储；自动召回方式 = **每步自动注入（还原 mihaji 原味）**。
  迁移属一次性行为：迁移完成后迁移代码**已整段移除**（用户：既然已迁完，就当它从未存在过），不再保留导入路径。

## 1. Hermes → DSH 绑定映射

| mihaji (Hermes) | 作用 | DSH 等价机制 | 状态 |
|---|---|---|---|
| `register_memory_provider` | 注册全局记忆 Provider | profile **host 服务**（bundle 插件注入，跨会话共享库句柄） | ✅ 已实现（apply 挂载全局工具/提示词） |
| `sync_turn` | 每轮后自动存 user 消息 | pre-step 抓 claimed 真实用户消息入库（§4.3） | ✅ 已实现 |
| `prefetch(query)` | 每步按当前输入检索并注入 | **`agent/pre-step` waterfall + `agent.ctx`**（§2） | ✅ 已实现 |
| `system_prompt_block()` | 提示词说明记忆库用法 | `systemPrompt.section()` | ✅ 已实现（order 100000） |
| `mihaji_memory` 工具 | search/remember/delete/count | `tools.register` 同构 tool | ✅ 已实现 |
| `on_session_end` | 收尾 | `session/disposed` | 可选，未做 |
| `store.py` / ChromaDB / BM25 | 存储与检索 | 进程内 q8 嵌入 + JSON（§5） | ✅ 已实现 |

## 2. 自动注入：机制验证结论（源码级）

DSH 每步流程（`dsh-agent-loop` `preStep`）：
```
assemble() 组 systemPrompt → 生成 runtime context 快照 →
waterfall "agent/pre-step"(payload.messages = 当前 step 用户消息) → decision.messages 逐条 append 进会话 → 进 model
```

- **`agent/pre-step` 是唯一能拿到「当前用户输入 query」做按需检索的挂点**；`systemPrompt.context` 的
  `AssembleContext` 只有 `scope/signal`，拿不到 query → 只能固定注入，不能还原 mihaji 的 `prefetch(query)`。
- **注入即持久化**：agent-loop 会把 `decision.messages` 逐条 `session.append("user/message", …)`（line 559）。
  所以注入要构造带 `source.kind:'plugin'` 的 UserMessage。
- **必须去重**：若每步都把 Top-N 记忆 append，会重复累积。需「保留上次注入快照、内容未变就不重复」——
  这正是 DSH `RuntimeContextProjection`（agent-loop 内置）的语义，移植时沿用同一思路。
- **import 约束**：动态 cordis 插件（cordis_define）跑在 `node:vm` 受限沙箱，**不能 `import`**，也访问不到
  `@deepseek-ai/dsh-llm` 的 `createUserMessage`。→ **正式实现必须是真实 profile bundle**（能 import DSH 包），
  不能靠动态插件手拼消息对象。

### 关键实证：agent/pre-step 是 scope 过滤的 waterfall，host 收不到，须挂 agent.ctx

注入链路经实机验证后锁定（`docs/dsh-port-design.md` 章节记录运行诊断）：
- **`agent/pre-step` 是 waterfall 事件，只派发给注册 ctx 携带该 agent scope 标签的 listener**。一个 untagged
  的 host/profile 行（即使 `{ global: true }`）**收不到** pre-step（实测：apply 挂载成功、但一次都不触发）。
- 相反，**emit 事件**（`agent/session-start`、`agent/status`、`session/event`）能到达 untagged host 行。
- 所以正确做法：host bundle 里 **扫描/监听 live agent**（`ctx.agents.list()` + `ctx.on('agent/created')` +
  `ctx.on('agent/session-start')`，均为 host 可达的 emit），对每个 agent 在 **`agent.ctx`**（其 scope 标签即该
  agent）上 `agent.ctx.on('agent/pre-step', handler)` —— 这才符合 pre-step 的 scope 准入。这是 DSH per-agent
  插件的标准机制（`dsh-time-context`、`dsh-file-reference-local` 同款）。
- 时序：恢复已有会话时 agent 可能晚于 host bundle apply 才出现 → 需 `list()` 立即扫 + `agent/created` 兜新 +
  短延迟重扫（800ms/3s）。
- **安装必须是物理 tarball，不能 `link:`**：`link:` 在 `web\node_modules` 只放符号链接，Node 按 realpath 回
  仓库源码目录解析不到 `@deepseek-ai`（它在 `profiles\node_modules\@deepseek-ai` 以 junction 形式存在）。物理
  解压在 `web\node_modules\dsh-mihaji` 内即可向上命中。
- **小改版重装的最优路径（实测）**：编辑 `C:\Users\miyaizu\.dsh\profiles\web\package.json` —— 把
  `dependencies["dsh-mihaji"]` 指向新 tgz（`file:C:/.../dsh-mihaji-<ver>.tgz`）并在 `dsh.profile.bundles`
  末尾保留 `"dsh-mihaji"`，然后 `cd profiles/web && pnpm install`（命中缓存秒装）。不要用 `dsh plugin rm+add`：
  它会整树重新解析/link transformers→onnxruntime-node 原生依赖（50+ 包、可能 120s 超时，还会因已删旧 tgz
  报 ENOENT）。装完重启 `dsh web`。

## 3. 推荐的 bundle 形态

- 命名：npm 包 `dsh-mihaji`，随本仓库分发（子目录 `dsh-mihaji/`）。
- 结构对齐 `dsh-whale-widget`（profile bundle 先例）：
  ```
  dsh-mihaji/
  ├── package.json          # name, type:module, main lib/index.js, dsh.bundle.patch → ./cordis.patch.yml
  ├── cordis.patch.yml      # insert 一行：{ id: dsh-mihaji, name: dsh-mihaji }
  ├── lib/
  │   └── index.js          # export { name, inject, apply }
  └── README.md
  ```
- 安装：物理 tarball 安装（不能 `link:`，原因见 §2）。小改版用 §2 的 package.json + `pnpm install` 路径。
  装完重启 `dsh web`（重启由用户手动执行并验收）。
- 移除：从 `web/package.json` 删掉依赖与 bundles 条目后 `pnpm install`（等价于 `dsh plugin rm`，且不残留引用）。

## 4. 功能切片（分步）

1. **注入切片（先验）**：`agent/pre-step` 拦截当前用户消息 → （占位）构造一条 plugin UserMessage 注入
   `decision.messages`，带快照去重 → 验证链路可通（需重启 web 后新会话观察）。
2. **自动记忆**：监听 `session/event`，在 `turn/end` 捕获 user 消息，应用 mihaji 噪音过滤
   （<20 字、`_GARBAGE_PREFIXES`），经存储 seam 落库。
3. **工具**：注册 `mihaji_memory`（search/remember/delete/count）。
4. **提示词**：注册 `systemPrompt.section` 说明记忆库与工具用法。
5. **可选 session_search**：注册 `session_search` 工具，走 `ctx.sessionQuery` 四模式
   （browse/read/scroll/discover），复用 Hermes `session_search_tool` 的调用约定但数据源为 DSH 会话。

## 5. 存储实现（已落地：进程内本地 q8 嵌入 + JSON 持久化）

候选 A（复用 Hermes ChromaDB，Python 子进程）太重易碎、B（纯 JS 无语义）缺召回，最终走 C 并迭代到现状：

- **C1 嵌入（0.5.0 起 = INT8 量化 q8）**：`lib/embed.js` 用 transformers.js 完全离线加载本地
  `paraphrase-multilingual-MiniLM-L12-v2`。默认加载量化版：首次启动从 Hermes HF 缓存快照的
  `onnx/model_quint8_avx2.onnx`（113MB）一次性异步拷贝到
  `$DSH_HOME/mihaji-memory/models/q8/onnx/model_quantized.onnx`（+ tokenizer/config 侧文件），
  之后每次直读 managed 目录；`dtype:'q8'` 找不到或加载失败时**自动回退 fp32**（Hermes 缓存
  `onnx/model.onnx` 或 `DSH_MIHAJI_MODEL` 覆盖）。冷启动降级纯关键词，模型就绪后后台回补缺 vec 行。
- **C2 检索**：双字 CJK/英文词关键词 + 语义余弦 → RRF 融合；`recall()` 用 strength² 加权 A-Res 采样 +
  chunk ±1 缝合；`isNearDuplicate`（规范化字符 Dice≥0.88 / 包含≥0.6）抑制自命中。
- **C3 持久化**：单 JSON 文件 `$DSH_HOME/mihaji-memory/memory.json`（version 2），每 chunk 自带 384 维
  Float 向量 → 启动无需重嵌。旧 Hermes `chroma.sqlite3` 只读一次性迁移已完成（1543 块 → 1545 条含后续
  自动入库），迁移代码已于 0.4.0 **整段删除**。

### 5.1 架构决策记录：进程内嵌入 vs 另起服务（2026-09，实测定案）

疑虑：fp32 MiniLM 常驻 ~1GB、「拖累 dsh 启动」。实测（同环境同代码路径，裸 node 进程内）：

| 变体 | 加载耗时 | 进程总 RSS | 相对空 node 增量 |
|---|---|---|---|
| fp32（0.4.0 及以前） | ~1.6s | ~773 MB（真实 profile 曾测 868 MB） | +707 ~ +798 MB |
| **q8（0.5.0 起，默认）** | ~1.1–1.6s | **~445–465 MB** | **+380 MB** |

精度（q8 vs fp32，10 句含中英）：逐句余弦一致性 mean **0.9944** / min 0.9921；top-3 近邻保持
**29/30=97%**（唯一偏差在低相似度尾部）；成对矩阵偏差 mean 0.009 / max 0.032。
→ 现有 1543 条 fp32 向量与新 q8 查询**混算安全**，无需重嵌。

结论（用户拍板 q8 进程内）：**另起服务不省总内存**（同样 ~380–450MB 只是换进程驻留，还多一个实体与
IPC），仅当「不想让 dsh web 进程变胖」或「多 profile 共享一嵌入守护」时才值得。当前取舍：`如无必要勿增实体`，
q8 进程内是终态。真实 dsh web 整进程（含全部 harness）RSS ≈ 664 MB，健康。

## 6. 待办 / 决策树

### 已完成（全部经真机验收，用户重启 `dsh web` 确认）

- [x] §4.1 注入链路 —— host 扫 agent（list + agent/created + session-start + 800ms/3s 重扫）→ `agent.ctx`
      挂 pre-step → 注入 plugin UserMessage，快照去重、claimed 过滤不自存。验收：消息顶部持续出现
      `## 相关回忆 🐾` 注入块（hits=5）。
- [x] §4.2 工具 —— `mihaji_memory`（search/remember/delete/count + recall/add/forget 兼容别名；
      strength 1–100 默认 70，memory_type/tags）经 `ctx.tools.register` 全局注册，全 agent 可见。
- [x] §4.3 自动记忆 —— pre-step 抓 claimed 真实用户消息 → 噪音过滤（长度/垃圾前缀/框架帧/元对话）
      → 入库 strength=20；auto-store 自身召回不回流。
- [x] §4.4 提示词 —— `systemPrompt.section('mihaji:memory', order=100000)` 说明记忆库与工具用法。
- [x] §5 存储 —— 进程内 q8 嵌入 + JSON 持久化落地（v0.5.0）；迁移完成且迁移代码删除（v0.4.0）。
- [x] §5.1 架构决策 —— q8 进程内（用户拍板），实测数据见上，另起服务被否决。
- [x] 收敛清理：`.gitignore` 排除 `dsh-mihaji/*.tgz`、`node_modules/`、`poc-embed/q8model/`、`*-vecs.json`；
      debug 日志门控统一为单一开关（`DSH_MIHAJI_DEBUG='0'` 静默全部 trace，默认开），v0.5.1 生效。

- [x] §4.5 `session_search`（v0.6.0）—— 单工具四形态（browse/discover/read/scroll），调用约定对齐 Hermes
      `session_search_tool.py`；数据源 = `ctx.sessionQuery`（DSH 会话历史）。**引擎决策**：不依赖 sqlite FTS
      后端（deployment 默认 `openAt:"never"` 时 `searchSessions/searchEvents` 抛
      `SESSION_QUERY_SEARCH_DISABLED`），改为用服务精确读原语（`listSessions`/`readSession`/`readEvent`/
      `readTitleSnapshots`）做**有界关键词扫描**——本机语料约 50 会话/14MB(zstd)，量级可行；排除
      `header.origin==='subagent'` 子代理会话；tool 事件视为噪音。单元测试
      `poc-embed/test-sessionsearch.mjs`（mock sessionQuery 全形态）。待 hina 真机验收。

版本沿革：0.1–0.2 注入链路 → 0.3 迁移/存储/工具/提示词 → **0.4.0** 删除 chroma 迁移代码 → **0.5.0** 默认
q8 量化嵌入 + 模型自动准备/回退 → **0.5.1** debug 门控统一 → **0.6.0** 新增 `session_search`。

### 未决 / 开放

- [ ] 睡眠巩固 cron：Hermes 版有 4am 离线巩固脚本（短内容过滤/Jaccard 去重/备份），尚未移植——用户未定
      是否要。
- [ ] §4.5 session_search 真机验收（hina 上实际调 browse/discover/scroll 各一次）。
- [ ] 发布准备：README 补 DSH 安装说明；仓库整体 git commit。
