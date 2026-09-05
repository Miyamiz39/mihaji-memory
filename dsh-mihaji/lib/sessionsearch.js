// dsh-mihaji — session_search: recall past DSH conversation sessions.
//
// DSH 移植版 session_search：单工具四形态，约定对齐 Hermes session_search_tool：
//   • browse    —— 无参数：最近的会话清单（id/标题/时间/cwd）
//   • discover  —— query：在历史会话全文里找最相关命中，按会话聚合返回
//                    (top 结果带 ±N 事件窗口；其余返回锚点行摘要)
//   • read      —— session_id：整段读取一个会话（过大时截头尾，指引 scroll）
//   • scroll    —— session_id + around_seq：以某个事件 seq 为中心取窗口
//
// 数据源 = ctx.sessionQuery（DSH 会话历史，live 优先）。引擎刻意不依赖 sqlite
// FTS 后端（deployment 默认 openAt:"never"，fulltext 被禁用），而是用服务自身的
// 精确读原语 listSessions / readSession / readEvent / readTitleSnapshots 做
// 有界关键词扫描。本机语料规模小（几十个会话），每次调用扫描最新 N 个会话即可。
// 约定过滤：排除 header.origin === 'subagent' 的子代理会话（同 Hermes 隐藏源）。

// ---- 轻量分词（与记忆库同款：CJK 双字 + 拉丁词） ----
function tokenize(text) {
  const out = new Set()
  const s = String(text || '').toLowerCase()
  const cjk = s.match(/[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]+/g) || []
  for (const run of cjk) {
    if (run.length === 1) out.add(run)
    for (let i = 0; i + 1 < run.length; i++) out.add(run.slice(i, i + 2))
  }
  const words = s.match(/[a-z0-9]+/g) || []
  for (const w of words) if (w.length >= 2) out.add(w)
  return out
}

const MIN_QUERY = 2

// 时间戳 → 本地 'YYYY-MM-DD HH:MM'
function fmtTime(ts) {
  if (!Number.isFinite(ts)) return ''
  const d = new Date(ts)
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

// 从事件里抽“人类可读正文”（user/assistant 的 text 块）。tool 事件默认视为噪音。
function eventText(event) {
  if (!event || !event.data) return ''
  const data = event.data
  switch (event.type) {
    case 'user/message':
      return messageText(data) // data = UserMessage
    case 'assistant/message':
      return messageText(data && data.message) // data = { turn, step, message }
    default:
      return ''
  }
}
function messageText(message) {
  if (!message || !Array.isArray(message.content)) return ''
  const parts = []
  for (const b of message.content) {
    if (b && b.type === 'text' && typeof b.text === 'string') parts.push(b.text)
  }
  return parts.join(' ')
}

function eventRole(event) {
  return event && event.type === 'user/message' ? 'user' : event && event.type === 'assistant/message' ? 'assistant' : ''
}

// 一行可读会话记录（scroll/read 输出用）。锚点事件加前缀标记。
// tool 事件等无 user/assistant 正文的事件返回空串，由 cleanLines 剔除。
function lineFor(event, { anchor = false, maxLen = 2000 } = {}) {
  const txt = eventText(event)
  if (!event || event.seq === undefined || !txt) return ''
  const text = txt.length > maxLen ? txt.slice(0, maxLen) + '…' : txt
  const when = fmtTime(event.time)
  const role = eventRole(event)
  return `[${anchor ? '►' : ''}${event.seq}] ${when} ${role}: ${text}`
}

function cleanLines(list) {
  return list.filter((l) => l && l.length > 0)
}

// ---- 扫描一个会话，返回 { bestScore, bestSeq, bestText, bestRole, bestTime, userCount } ----
function scanSession(log, qf) {
  const events = Array.isArray(log) ? log : []
  let bestScore = 0
  let best = null
  let userCount = 0
  for (const ev of events) {
    const txt = eventText(ev)
    if (!txt) continue
    if (ev.type === 'user/message') userCount++
    const hits = new Set()
    for (const f of qf) if (txt.includes(f)) hits.add(f)
    if (hits.size === 0) continue
    let score = 0
    for (const f of hits) score += f.length >= 2 ? 2 : 1
    if (score > bestScore) {
      bestScore = score
      best = {
        seq: ev.seq,
        text: txt,
        role: eventRole(ev),
        time: ev.time,
      }
    }
  }
  return { bestScore, best, userCount }
}

const HIDDEN_ORIGIN = 'subagent'

// 每个请求一个实例：只构造一次扫描上下文。
export function createSessionSearch({ sq, trace = () => {} } = {}) {
  async function listCandidates() {
    const records = await sq.listSessions()
    if (!Array.isArray(records)) return []
    const out = []
    for (const r of records) {
      const h = r && r.header
      if (!h || !h.id) continue
      if (h.origin === HIDDEN_ORIGIN) continue
      out.push({ record: r, header: h })
    }
    out.sort((a, b) => (b.header.createdAt || 0) - (a.header.createdAt || 0))
    return out
  }

  async function titleOf(id) {
    try {
      const t = await sq.readTitle(id)
      return t && t.title ? t.title : null
    } catch {
      return null
    }
  }

  // ---- browse：最近的会话（只读 header + 标题，不解压正文） ----
  async function browse(limit) {
    const cands = await listCandidates()
    const picked = cands.slice(0, limit)
    const rows = []
    for (const c of picked) {
      rows.push({
        session_id: c.header.id,
        title: await titleOf(c.header.id),
        created_at: fmtTime(c.header.createdAt),
        cwd: c.header.cwd || null,
        agent_preset: c.header.agentPreset || null,
        live: !!c.record.live,
      })
    }
    return {
      success: true,
      mode: 'browse',
      results: rows,
      count: rows.length,
      hint:
        '浏览的是最近会话。要搜内容用 query=；要整段读用 session_id=；要翻一段用 session_id + around_seq。',
    }
  }

  // ---- read：整段读一个会话（大会话截头尾） ----
  async function readSession(sessionId, { head = 25, tail = 12 } = {}) {
    if (!sq.readSession) throw new Error('sessionQuery.readSession 不可用')
    const snap = await sq.readSession(sessionId)
    const events = snap && Array.isArray(snap.events) ? snap.events : []
    const header = (snap && snap.session) || {}
    const total = events.length
    const truncated = total > head + tail
    const slice = truncated ? events.slice(0, head).concat(events.slice(total - tail)) : events
    const lines = cleanLines(slice.map((ev) => lineFor(ev)))
    const out = {
      success: true,
      mode: 'read',
      session_id: sessionId,
      session_meta: {
        created_at: fmtTime(header.createdAt),
        cwd: header.cwd || null,
        agent_preset: header.agentPreset || null,
        parent_session: header.parentSession || null,
      },
      message_count: total,
      truncated,
      messages: lines,
    }
    if (truncated) {
      out.hint = `该会话共 ${total} 条可读事件，展示前 ${head} + 后 ${tail} 条。要用 around_seq 翻中间：先 discover(query=...) 拿锚点 seq，或 scroll(session_id, around_seq=<任意 seq>)。`
    }
    return out
  }

  // ---- scroll：以某 seq 为中心取窗口 ----
  async function scroll(sessionId, seq, window) {
    if (!sq.readEvent) throw new Error('sessionQuery.readEvent 不可用')
    const w = await sq.readEvent({ sessionId, seq, before: window, after: window })
    const events = Array.isArray(w.events) ? w.events : []
    const lines = cleanLines(
      events.map((ev) => lineFor(ev, { anchor: ev && ev.seq === seq })),
    )
    return {
      success: true,
      mode: 'scroll',
      session_id: sessionId,
      around_seq: seq,
      window,
      messages: lines,
      hint:
        '向前翻：around_seq 设为返回最后一条的 seq；向后翻：设为第一条的 seq（边界条会重复作参照）。窗口两端若已到会话边界，说明到底了。',
    }
  }

  // ---- discover：关键词扫描 + 会话聚合 ----
  async function discover(query, { limit = 3, sort = null, detail = 'adaptive', scanLimit = 60 } = {}) {
    const q = String(query || '').trim()
    if (q.length < MIN_QUERY) {
      return { success: false, mode: 'discover', error: 'query 太短，无法搜索。' }
    }
    const qf = tokenize(q)
    const cands = await listCandidates()
    const scanned = cands.slice(0, scanLimit)

    const hits = [] // { header, record, bestScore, best }
    for (const c of scanned) {
      let log
      try {
        const snap = await sq.readSession(c.header.id)
        log = snap && snap.events
      } catch {
        continue // 损坏/不可读的会话跳过
      }
      const r = scanSession(log, qf)
      if (r.bestScore > 0) hits.push({ header: c.header, record: c.record, ...r })
    }

    if (hits.length === 0) {
      return {
        success: true,
        mode: 'discover',
        query: q,
        detail,
        results: [],
        count: 0,
        sessions_searched: scanned.length,
        message: '没有找到匹配的会话。换个关键词试试。',
      }
    }

    // 相关度排序；sort=newest/oldest 只在同分时按时间微调。
    hits.sort((a, b) => {
      if (b.bestScore !== a.bestScore) return b.bestScore - a.bestScore
      const ta = a.header.createdAt || 0
      const tb = b.header.createdAt || 0
      return sort === 'oldest' ? ta - tb : tb - ta
    })
    const picked = hits.slice(0, limit)

    const results = []
    for (let i = 0; i < picked.length; i++) {
      const p = picked[i]
      const full = detail === 'full' || (detail !== 'full' && i === 0)
      const entry = {
        session_id: p.header.id,
        created_at: fmtTime(p.header.createdAt),
        cwd: p.header.cwd || null,
        title: await titleOf(p.header.id),
        matched_role: p.best ? p.best.role : null,
        match_seq: p.best ? p.best.seq : null,
        matched_at: p.best ? fmtTime(p.best.time) : null,
        snippet: p.best ? (p.best.text.length > 300 ? p.best.text.slice(0, 300) + '…' : p.best.text) : '',
        detail: full ? 'full' : 'compact',
      }
      if (full && p.best) {
        try {
          const w = await sq.readEvent({ sessionId: p.header.id, seq: p.best.seq, before: 2, after: 3 })
          const events = Array.isArray(w.events) ? w.events : []
          entry.messages = cleanLines(events.map((ev) => lineFor(ev, { anchor: ev && ev.seq === p.best.seq, maxLen: 1500 })))
        } catch {
          entry.messages = []
        }
      }
      results.push(entry)
    }

    return {
      success: true,
      mode: 'discover',
      query: q,
      detail,
      results,
      count: results.length,
      sessions_searched: scanned.length,
      hint:
        '想深入某条：read(session_id=...) 整段读；scroll(session_id, around_seq=<match_seq>) 看命中上下文。',
    }
  }

  // 主入口：按参数推断形态（对齐 Hermes 调用约定）
  async function run(args = {}) {
    const rawSeq = args.around_seq ?? args.around_message_id
    const seqNum = rawSeq === undefined || rawSeq === null || rawSeq === '' ? undefined : Number(rawSeq)
    const hasSession = typeof args.session_id === 'string' && args.session_id.trim().length > 0
    const sessionId = hasSession ? args.session_id.trim() : undefined

    // scroll 优先：显式锚点
    if (sessionId && Number.isFinite(seqNum)) {
      const w = Math.max(1, Math.min(20, Number.isFinite(args.window) ? args.window : 5))
      return scroll(sessionId, seqNum, w)
    }
    // read
    if (sessionId) return readSession(sessionId)
    // browse
    const query = String(args.query || '').trim()
    if (!query) {
      const n = Math.max(1, Math.min(20, Number.isFinite(args.limit) ? args.limit : 10))
      return browse(n)
    }
    // discover
    const limit = Math.max(1, Math.min(10, Number.isFinite(args.limit) ? args.limit : 3))
    const sort = args.sort === 'oldest' ? 'oldest' : args.sort === 'newest' ? 'newest' : null
    const detail = args.detail === 'full' ? 'full' : 'adaptive'
    return discover(query, { limit, sort, detail })
  }

  return { run }
}
