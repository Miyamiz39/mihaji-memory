// dsh-mihaji — Mihaji Memory for DeepSeek Harness.
//
// Slice #3 — persistent semantic memory:
//   • persistent store (JSON under DSH_HOME) + 300/50 chunking,
//   • fully-offline local MiniLM embeddings (transformers.js) for semantic recall,
//   • hybrid keyword+vector search (RRF) with mihaji-style weighted recall,
//   • per-step auto recall injection (wiring proven in Slices #1/#2),
//   • global `mihaji_memory` tool + auto-memorization + a system-prompt section.
//
// ARCHITECTURE (see repo docs): `agent/pre-step` is a scope-filtered WATERFALL
// delivered only to listeners registered on the dispatching agent's scoped ctx
// (`agent.ctx`). A host/profile row is NOT admissible. So for every live agent
// we install the pre-step handler ON `agent.ctx`. Its `messages` argument is the
// inbox-claimed GENUINE user input; our injected recall only lives in the
// returned `decision.messages`, so auto-memorization never re-stores it.
//
// Tool + prompt section register on the global layer from this host row, visible
// to every agent (same mechanism dsh-tool-todo uses). This is a physical profile
// bundle (tarball under profiles\web\node_modules) so @deepseek-ai/* and
// @huggingface/* resolve by upward walk.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { createStore } from './store.js'
import { createSessionSearch } from './sessionsearch.js'

const name = 'dsh-mihaji'
// `tools` is guaranteed on the profile ctx; waiting on it also orders us after
// the registry mounts. Everything else is optional and read via ctx.get(...).
const inject = ['tools']

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

const DEBUG_LOG = path.join(
  process.env.DSH_HOME || path.join(os.homedir(), '.dsh'),
  'mihaji-debug.log',
)

function debug(line) {
  try {
    fs.appendFileSync(DEBUG_LOG, `[${new Date().toISOString()}] ${line}\n`)
  } catch { /* never take the loop down */ }
}

function textOf(message) {
  if (!message || !Array.isArray(message.content)) return ''
  if (message.content.length === 1 && message.content[0] && message.content[0].type === 'text') {
    return typeof message.content[0].text === 'string' ? message.content[0].text : ''
  }
  return ''
}

// ---------------------------------------------------------------------------
// Auto-memorization noise filter (mirrors mihaji sync_turn semantics).
// Auto-memory should record USER's durable facts/preferences, not harness
// scaffolding (goal_round frames, objective text, system reminders, injected
// recall) or meta chit-chat about the plugin itself.
// ---------------------------------------------------------------------------

const AUTO_MIN_LEN = 12
const AUTO_GARBAGE = [
  'Review the conversation above',
  '更新 two things',
  'skill library',
  // Harness / agent-loop control frames that arrive as "user" turns.
  '<goal_round>',
  'Continue working toward the objective',
  'Objective: "',
  'Round:',
  'system-reminder',
  'This is an automatically generated checkpoint',
  'Treat the captured context as established background',
  '## 相关回忆 🐾', // injected recall must never be auto-memorized
  // Meta talk about building/testing mihaji itself.
  '我们直接推进',
  '我先做',
  '做一个 dsh 插件',
  '把 mihaji-memory 改造',
]

function isAutoStorable(text) {
  const stripped = String(text || '').trim()
  if (stripped.length < AUTO_MIN_LEN) return false
  for (const p of AUTO_GARBAGE) if (stripped.includes(p)) return false
  return true
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

function buildMemoryTool(store, trace) {
  const resultText = (s) => ({ type: 'text', text: s })
  return defineTool({
    name: 'mihaji_memory',
    description:
      '语义记忆库 — 混合检索（本地嵌入语义 + 关键词）找回过去的对话、约定、个人事实，或主动存储长期偏好。\n' +
      'ACTIONS:\n' +
      '• search — 按关键词或语义搜索过往记忆，返回最相关片段（可用 query）。\n' +
      '• remember — 主动存储一条重要记忆（用 text，可指定 strength/memory_type/tags）。\n' +
      '• delete — 删除与 query 匹配的记忆。\n' +
      '• count — 查看记忆库条目数。\n' +
      '注意：回复前会自动注入相关回忆到上下文；需要补充查询时用 search。',
    parameters: {
      action: {
        type: 'string',
        required: true,
        enum: ['search', 'remember', 'delete', 'count', 'recall', 'add', 'forget'],
        description: '要执行的动作。search/remember/delete/count 为标准动作；recall/add/forget 为兼容别名。',
      },
      query: { type: 'string', description: "action='search'/'delete' 时的关键词。" },
      text: { type: 'string', description: "action='remember' 时要存储的记忆内容（必填）。" },
      limit: { type: 'integer', description: '返回条数（search 默认 5）。' },
      strength: { type: 'integer', description: '记忆强度 1-100（remember 可选，默认 70）。' },
      memory_type: { type: 'string', description: 'knowledge/preference/event/task/general（remember 可选，默认 general）。' },
      tags: { type: 'string', description: '逗号分隔标签（remember 可选）。' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          text: { type: 'string', required: true },
        },
      },
      render: (_args, value) => [resultText(value.text)],
    },
    async execute(args) {
      const action = args.action
      try {
        if (action === 'search' || action === 'recall') {
          const query = String(args.query || '').trim()
          if (!query) return { text: 'search 需要 query 参数。' }
          const limit = Math.max(1, Math.min(20, Number.isFinite(args.limit) ? args.limit : 5))
          const results = await store.hybrid(query, { limit })
          if (results.length === 0) return { text: '没找到相关记忆。' }
          const body = results
            .map((r) => `[${r.created_at}] ${r.content} (strength=${r.strength}, ${r.memory_type})`)
            .join('\n')
          return { text: `找到 ${results.length} 条相关记忆：\n${body}` }
        }
        if (action === 'remember' || action === 'add') {
          const text = String(args.text || '').trim()
          if (!text) return { text: 'remember 需要 text 参数。' }
          const strength = Math.max(1, Math.min(100, Number.isFinite(args.strength) ? args.strength : 70))
          const memoryType = args.memory_type || args.type || 'general'
          const n = await store.add(text, { strength, memoryType, tags: args.tags })
          trace(`tool: remembered (${n} chunk): ${text.slice(0, 40)}`)
          return { text: `已记住（${n} 条）：${text}` }
        }
        if (action === 'delete' || action === 'forget') {
          const query = String(args.query || '').trim()
          if (!query) return { text: 'delete 需要 query 参数。' }
          const n = await store.deleteByQuery(query)
          return { text: n > 0 ? `已删除 ${n} 条匹配记忆。` : '没有找到匹配的记忆可删除。' }
        }
        if (action === 'count') {
          return { text: `记忆库现有 ${store.count()} 条片段。` }
        }
        return { text: `未知 action: ${action}` }
      } catch (e) {
        return { text: `mihaji_memory 出错: ${e && e.message ? e.message : String(e)}` }
      }
    },
    presentCall: (args) => ({
      card: 'generic',
      title: `mihaji_memory · ${String(args.action || '')}`,
      kind: 'other',
      rawInput: args,
    }),
  })
}

// session_search — recall DSH conversation history via ctx.sessionQuery. Single
// tool, four shapes inferred from args (Hermes session_search_tool parity):
//   query        → discover  (search past sessions, top hit hydrated ±window)
//   session_id   → read      (whole session, head/tail bounded when huge)
//   session_id + around_seq → scroll (window centred on one event)
//   no args      → browse    (recent sessions)
function buildSessionSearchTool(sessionSearch, trace) {
  const resultText = (s) => ({ type: 'text', text: s })
  return defineTool({
    name: 'session_search',
    description:
      '翻查过往对话会话（DSH 会话历史）。四种形态按参数推断：\n' +
      '• browse — 不带参数：列出最近的会话。\n' +
      '• discover — 传 query：在所有历史会话里搜索关键词，按会话聚合返回命中（默认 top-1 带上下文窗口）。\n' +
      '• read — 只传 session_id：整段读取某个会话。\n' +
      '• scroll — session_id + around_seq（会话内事件序号）：以某条事件为中心翻窗口。\n' +
      '适合回答「我们之前聊过 X 吗 / 上次怎么说的 / 之前在哪讨论过」。返回 JSON。',
    parameters: {
      query: { type: 'string', description: 'discover 形态的搜索关键词。省略且无 session_id 时为 browse。' },
      session_id: { type: 'string', description: 'read/scroll 形态的目标会话 id（discover 结果里的 session_id）。' },
      around_seq: { type: 'integer', description: 'scroll 形态：窗口中心的事件 seq（discover 返回的 match_seq，或窗口边界 seq）。' },
      limit: { type: 'integer', description: 'discover 最多返回会话数 / browse 最近会话数（默认 3，上限 10；browse 上限 20）。' },
      sort: { type: 'string', enum: ['newest', 'oldest'], description: 'discover 同分时的时效偏向：省略=相关度优先，newest=近期优先，oldest=早期优先。' },
      detail: { type: 'string', enum: ['adaptive', 'full'], description: "discover 详情：adaptive（默认，仅 top-1 带上下文窗口），full（每个结果都带窗口）。" },
      window: { type: 'integer', description: 'scroll 的窗口半径（1-20，默认 5）。' },
      role_filter: { type: 'string', description: '预留兼容：本版固定只搜 user/assistant 正文（工具输出视为噪音）。' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          text: { type: 'string', required: true },
        },
      },
      render: (_args, value) => [resultText(value.text)],
    },
    async execute(args) {
      const t0 = Date.now()
      try {
        const out = await sessionSearch.run(args || {})
        trace(`tool: session_search done in ${Date.now() - t0}ms (mode=${out && out.mode})`)
        return { text: JSON.stringify(out, null, 2) }
      } catch (e) {
        trace(`tool: session_search FAILED: ${e && e.message ? e.message : String(e)}`)
        return {
          text: JSON.stringify({
            success: false,
            error: e && e.message ? e.message : String(e),
          }, null, 2),
        }
      }
    },
    presentCall: (args) => ({
      card: 'generic',
      title: `session_search${args && args.query ? ` · ${String(args.query).slice(0, 40)}` : args && args.session_id ? ' · read/scroll' : ' · browse'}`,
      kind: 'other',
      rawInput: args,
    }),
  })
}

function apply(ctx) {
  // Single honest gate: DSH_MIHAJI_DEBUG='0' silences the whole trace (store +
  // apply + wiring); default is ON so diagnostics stay visible during dev.
  const debugEnabled = process.env.DSH_MIHAJI_DEBUG !== '0'
  const trace = debugEnabled ? debug : () => {}
  const store = createStore({ trace })
  trace(`apply: bundle mounted (${name}); existing chunks=${store.count()}; embedReady=${store.isEmbedReady()}`)

  // ---- global tool ----
  try {
    ctx.tools.register(buildMemoryTool(store, trace))
    trace('apply: registered mihaji_memory tool on ctx.tools')
  } catch (e) {
    trace(`apply: tool registration failed: ${e && e.message ? e.message : String(e)}`)
  }

  // ---- session_search tool (optional: needs ctx.sessionQuery) ----
  const sq = ctx.get('sessionQuery')
  if (sq) {
    try {
      const sessionSearch = createSessionSearch({ sq, trace })
      ctx.tools.register(buildSessionSearchTool(sessionSearch, trace))
      trace('apply: registered session_search tool on ctx.tools')
    } catch (e) {
      trace(`apply: session_search init failed: ${e && e.message ? e.message : String(e)}`)
    }
  } else {
    trace('apply: sessionQuery absent — session_search not registered')
  }

  // ---- system-prompt section ----
  const systemPrompt = ctx.get('systemPrompt')
  if (systemPrompt && typeof systemPrompt.section === 'function') {
    try {
      systemPrompt.section({
        name: 'mihaji:memory',
        order: 100000,
        text: () => {
          const n = store.count()
          const embed = store.isEmbedReady()
          const head = n === 0
            ? '# Mihaji 记忆库 🐾\n记忆库还是空的。'
            : `# Mihaji 记忆库 🐾\n已存 ${n} 条记忆片段。`
          return `${head}\n` +
            (embed ? '' : '(语义模型仍在加载，当前为关键词召回)\n') +
            '会话中出现的 "## 相关回忆 🐾" 块是系统自动召回的长期记忆上下文，用于辅助回答，' +
            '不是用户的新指令，不要把它当成需要回应的提问。\n' +
            '需要时用 `mihaji_memory` 工具：search（检索）/ remember（主动存重要事实或偏好）/ delete（删除）/ count（查看条目数）。\n' +
            '要翻查「过去的完整对话/会话」用 `session_search` 工具：不带参数浏览最近会话、query= 搜索内容、' +
            'session_id 读整段、session_id+around_seq 翻窗口（返回 JSON）。'
        },
      })
      trace('apply: registered systemPrompt section mihaji:memory')
    } catch (e) {
      trace(`apply: systemPrompt section failed: ${e && e.message ? e.message : String(e)}`)
    }
  } else {
    trace('apply: systemPrompt service not present; skipping section')
  }

  // ---- per-agent wiring ----
  const installedOn = new Set()
  const lastRecallPerAgent = new Map()
  const rememberedTurns = new Set()

  function makePreStepHandler(agent) {
    const agentId = agent ? String(agent.id || '') : ''
    return async ({ messages, turn, step }, next) => {
      try {
        const decision = await next()
        if (!decision || decision.kind !== 'enter') return decision
        const stepIsFirst = step === undefined || step <= 1

        // auto-memorize genuine user turns on the first step of a turn
        if (stepIsFirst && Array.isArray(messages)) {
          for (const claimed of messages) {
            const txt = textOf(claimed)
            if (!txt) continue
            const key = `${agentId}:${turn}`
            if (rememberedTurns.has(key)) break
            if (isAutoStorable(txt)) {
              try {
                await store.add(txt, { strength: 20, memoryType: 'general' })
                rememberedTurns.add(key)
                trace(`auto-store: agent=${agentId} turn=${turn} stored (${txt.slice(0, 30)})`)
              } catch {
                // never break the loop on store failure
              }
            }
          }
        }

        // recall injection
        const lastClaimed = messages && messages.length ? messages[messages.length - 1] : undefined
        const query = lastClaimed ? textOf(lastClaimed).trim() : ''
        if (!query) return decision

        let hits = []
        try {
          hits = await store.recall(query, { limit: 5 })
        } catch {
          hits = []
        }
        const recallLines = hits.filter((r) => r && r.content).slice(0, 5)
          .map((r) => `- [${r.created_at || ''}] ${r.content}`)
        if (recallLines.length === 0) return decision

        const snapshot = ['## 相关回忆 🐾', ...recallLines].join('\n')
        if (lastRecallPerAgent.get(agentId) === snapshot) return decision
        if (decision.messages.some((m) => textOf(m) === snapshot)) return decision

        lastRecallPerAgent.set(agentId, snapshot)
        trace(`pre-step: injecting recall (agent=${agentId}, hits=${recallLines.length})`)

        const injected = createUserMessage({
          content: [{ type: 'text', text: snapshot }],
          source: { kind: 'plugin', plugin: 'dsh-mihaji', form: 'recall' },
        })
        const lastClaimedIndex = decision.messages.findLastIndex((m) => messages.includes(m))
        if (lastClaimedIndex >= 0) {
          return { ...decision, messages: decision.messages.toSpliced(lastClaimedIndex + 1, 0, injected) }
        }
        return { ...decision, messages: [...decision.messages, injected] }
      } catch (e) {
        trace(`pre-step: ERROR ${e && e.message ? e.message : String(e)}`)
        try { return await next() } catch { return { kind: 'enter', messages: (messages && messages.slice()) || [] } }
      }
    }
  }

  function wire(agent) {
    if (!agent || installedOn.has(agent)) return
    const actx = agent.ctx
    if (!actx || typeof actx.on !== 'function') return
    installedOn.add(agent)
    trace(`wire: agent=${String(agent.id || '')} -> register pre-step on agent.ctx`)
    actx.on('agent/pre-step', makePreStepHandler(agent))
  }

  const scan = () => {
    const agents = ctx.get('agents')
    if (!agents || typeof agents.list !== 'function') return
    for (const a of agents.list()) wire(a)
  }

  scan()
  ctx.on('agent/created', ({ agent }) => wire(agent), { global: true })
  ctx.on('agent/session-start', ({ agent }) => wire(agent), { global: true })
  ctx.on('agent/disposed', ({ agent }) => {
    if (agent) {
      installedOn.delete(agent)
      lastRecallPerAgent.delete(String(agent.id || ''))
    }
  }, { global: true })

  const rescans = [800, 3000].map((ms) => setTimeout(scan, ms))

  ctx.effect(() => {
    for (const t of rescans) clearTimeout(t)
    installedOn.clear()
    lastRecallPerAgent.clear()
    rememberedTurns.clear()
  })

  void store
}

export { name, inject, apply }
