# dsh-mihaji — Mihaji Memory for DeepSeek Harness

把 **mihaji-memory**（Hermes 时代的跨会话语义记忆）移植为 **DeepSeek Harness (DSH) bundle**。
该包安装在 Profile 中供模块解析，但插件行只应由 `hina` agent preset 挂载，因此工具、提示词与自动召回都只进入 hina 的作用域。它提供：

- **持久语义记忆**：`memory.json` 单文件持久化，300/50 分块，双字 CJK + 拉丁词关键词与语义双轨检索（RRF 融合）
- **本地 q8 量化嵌入**：transformers.js 离线加载 MiniLM（INT8，进程内 ~450MB，精度与 fp32 一致性 ≥0.992），模型文件首次自动从 HuggingFace 缓存准备、失败自动回退 fp32
- **每步自动召回**：按当前输入注入相关记忆（`## 相关回忆 🐾`），快照去重、claimed 过滤不自存
- **自动记忆**：真实用户消息经噪音过滤后自动入库（strength=20）
- **启动清理**：插件启动加载记忆库时 为首次发现的 `strength <= 0` 记忆记录归零时间 并删除归零已满 7 天的记忆；不启后台定时器 缺失或非法归零时间不删
- **`mihaji_memory` 工具**：search / remember / delete / count
- **`session_search` 工具**（v0.6.0）：翻查 DSH 会话历史，四形态 browse / discover(query) / read(session_id) / scroll(session_id+around_seq)，调用约定对齐 Hermes `session_search_tool.py`

## 安装与挂载

```bash
# 1. 打包
cd dsh-mihaji && npm pack            # 产出 dsh-mihaji-<ver>.tgz

# 2. 将 tarball 安装为目标 Profile 的 dependency，但不要加入
#    dsh.profile.bundles；Host/Profile 不应全局挂载它。
cd <DSH_HOME>/profiles/web && pnpm install

# 3. 在 <DSH_HOME>/.agent-presets/hina/agent.cordis.yml 中挂载：
# - id: mihaji
#   name: dsh-mihaji

# 4. 重启 Profile
```

> 必须物理 tarball 安装（不能 `link:`），否则 `@deepseek-ai/*` 解析不到。
> Mihaji 由 hina preset 挂载后，其工具、提示词和 `agent/pre-step` 自动召回自然只对 hina 生效，不需要检查 preset id。
> 数据与模型位于 `<DSH_HOME>/mihaji-memory/`（memory.json、models/q8），跨 profile 共享。
> 调试日志：`<DSH_HOME>/mihaji-debug.log`，设 `DSH_MIHAJI_DEBUG=0` 静默。

## 布局

```
dsh-mihaji/
├── package.json          # name dsh-mihaji / dsh.bundle.patch → cordis.patch.yml
├── cordis.patch.yml
└── lib/
    ├── index.js          # 插件主体：工具注册、pre-step 召回注入、自动记忆
    ├── store.js          # 记忆库：JSON 持久化 + 关键词/语义混合检索
    ├── embed.js          # q8/fp32 本地嵌入 + 模型一次性准备与回退
    ├── chunk.js          # 300/50 分块
    └── sessionsearch.js  # session_search：DSH 会话历史检索（v0.6.0）
```

架构与决策记录见仓库根 `docs/dsh-port-design.md`。MIT License。
