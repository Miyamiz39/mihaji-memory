# Mihaji Memory 改进工作区

从 Hermes 当前激活的 `mihaji` 插件复制而来的**改进工作副本**。
原版位置：`~/AppData/Local/hermes/profiles/hina/plugins/mihaji/`

---

## ⚠️ 这是工作副本，不是活跃插件

Hermes 加载的是**原版路径**下的插件。本目录改动**不会**自动生效。

任何想回原版/合并的改动，都需要手动复制回去（见底部"回退 / 合并"）。

---

## 本次改动（2026-08-30 v1.2.0）

完成了系统性代码审查、致命缺陷修复与第二轮体验进阶优化：

| 文件 | 改动 |
|---|---|
| `store.py` | 修复 `recall()` strength 采样 Bug 与离线模型加载；新增 `delete()` 方法；新增 `get_stitched_context()` 上下文缝合；支持 BM25 磁盘缓存与增量追加 |
| `hybrid_search.py` | 修复 RRF 融合元数据透传；升级 A-Res 加权采样算法；支持 `BM25Index` 增量 `add_documents` / `remove_documents` 与磁盘持久化 |
| `__init__.py` | 精简 Tool Schema（去除 `add` / `recall` 冗余，聚焦 `search` / `remember` / `delete` / `count`）；`prefetch` 集成上下文缝合 |
| `maintenance.py` | 修复 import 路径与跨长度桶 Jaccard 去重；增加低强度/陈旧记忆清理支持与 dry-run 仿真 |
| `plugin.yaml` | 升级版本为 `1.2.0`，补充依赖声明与 Hooks 声明 |
| `requirements.txt` | 运行时依赖清单（含 torch 版本与镜像提示） |
| `AGENTS.md` | 项目交接入门 + 历史审计记录 + 修订日志 |

---

## 依赖清单与版本约束

| 包 | 约束 | 实测版本 | 用途 |
|---|---|---|---|
| `chromadb` | `>=1.5.0,<2.0` | 1.5.9 | 向量数据库 + 持久化 |
| `jieba` | `>=0.42,<1.0` | 0.42.1 | 中文分词（BM25 前置） |
| `rank_bm25` | `>=0.2.2,<0.3` | 0.2.2 | BM25 关键词检索 |
| `sentence-transformers` | `>=5.0,<7.0` | 6.0.0 | 多语言嵌入模型 |
| `pyyaml` | `>=6.0` | 6.0.3 | config.yaml 解析 |

> **依赖关系不可降级**：`jieba` 和 `rank_bm25` 是 `hybrid_search.py` 的核心 import，
> 不是可选增强——混合检索（BM25+向量 RRF 融合）是 mihaji 的卖点，不是装饰品。

---

## 安装

### 在 Hermes 已有 venv 里装（推荐）

```bash
cd ~/AppData/Local/hermes
source venv/Scripts/activate   # Windows Git Bash
uv pip install -r ~/Desktop/mihaji-work/requirements.txt
```

### 全新环境

```bash
# 1. 装 Hermes（参照官方文档）
# 2. 激活 venv 后执行上面那条命令
# 3. sentence-transformers 首次会下载模型，
#    mihaji 默认用的是 paraphrase-multilingual-MiniLM-L12-v2 (~470MB)
```

### torch 选 CPU 还是 GPU

`sentence-transformers` 会拉 `torch>=2.2`，请按需：

- **仅 CPU**：体积小，启动慢但够用
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- **有 NVIDIA GPU**：默认 `pip install torch` 即可（CUDA 版本由 pip 自动选）

---

## 回退 / 合并到原版

### 改坏了？回退到原版

Hermes 一直在用原版，本目录改了啥都不影响线上。所以——

- **"线上坏了"**：跟本目录无关，原版没动过；查 hermes 配置或依赖问题
- **"想看本目录改得对不对"**：在本目录直接 `python -c "import chromadb, jieba, rank_bm25; print('ok')"` 验证

### 改完了想合并到原版

```bash
# 1. 先停 Hermes（避免文件占用）
# 2. 备份原版
cp -r ~/AppData/Local/hermes/profiles/hina/plugins/mihaji \
      ~/AppData/Local/hermes/profiles/hina/plugins/mihaji.bak.$(date +%Y%m%d)

# 3. 仅复制改动过的声明文件（推荐，不要无脑覆盖整个目录）
cp ~/Desktop/mihaji-work/plugin.yaml \
   ~/AppData/Local/hermes/profiles/hina/plugins/mihaji/plugin.yaml

# 4. requirements.txt 不用合并——它是给人类看的，原版 plugin 目录不需要它
#    （如果想保留，复制过去也行，无副作用）

# 5. 重启 Hermes
```

---

## 工作流建议

1. 改 `plugin.yaml` / 源码前先在本目录改完 review
2. 复杂改动建议走 git（`cd ~/Desktop/mihaji-work && git init`）
3. 合并前先 diff 原版 `plugin.yaml`，确认只多了 `dependencies` 字段
4. 涉及 Python 源码改动时，本地必须有可用的 venv 测过 `python -c "import store"`

---

## 文件清单

```
mihaji-work/
├── HANDOFF.md             ← 给接手者：项目入门 + bug 清单（**先读这个再动手**）
├── README.md              ← 本文件：改进说明 + 回退/合并指引
├── requirements.txt       ← pip/uv 安装清单
├── plugin.yaml            ← 插件元数据 + 依赖声明
├── __init__.py            ← 未改动（Hermes 入口）
├── store.py               ← 未改动（ChromaDB 封装）
├── hybrid_search.py       ← 未改动（BM25 + RRF）
└── maintenance.py         ← 未改动（CLI 维护脚本）
```