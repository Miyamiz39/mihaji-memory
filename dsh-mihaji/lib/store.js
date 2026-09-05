// dsh-mihaji — persistent semantic memory store.
//
// Storage is a single JSON file under $DSH_HOME/mihaji-memory/memory.json,
// shared by all sessions in this profile host, so memory survives restarts.
// Each chunk row carries its own 384-dim vector (Float array) so we never need
// to re-embed the whole library at boot.
//
// Retrieval is hybrid, mirroring mihaji:
//   • semantic leg   — cosine over local-MiniLM embeddings (optional; if the
//                      model is cold/absent we fall back to keyword only)
//   • keyword leg    — CJK-bigram + latin-word overlap
//   • Reciprocal-Rank Fusion merges both ranked lists
//   • recall() does a relevance top-N + strength²-weighted serendipity draw,
//     exactly like Hermes MihajiMemoryProvider.prefetch().
//
// The store never blocks the agent loop on embedding: embed warmup runs in the
// background and search/recall just return whatever quality is currently
// achievable (keyword-only until the model is ready).

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createEmbedder, cosine } from './embed.js'
import { chunkText, stitchChunks } from './chunk.js'

const CHUNK_SIZE = 300
const CHUNK_OVERLAP = 50

function storeDir() {
  const home = process.env.DSH_HOME || path.join(os.homedir(), '.dsh')
  return path.join(home, 'mihaji-memory')
}
function storeFile() {
  return path.join(storeDir(), 'memory.json')
}

// ---- lightweight tokenization + keyword similarity (pure JS) ----
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
function keywordScore(q, doc) {
  let score = 0
  for (const f of q) if (doc.has(f)) score += f.length >= 2 ? 2 : 1
  return score
}

// Normalize-away whitespace/punct char Dice — used to drop self-hit and near-dup
// entries (a whole sentence auto-stored earlier must not re-hit its own re-paste).
function normChars(s) {
  return String(s || '').toLowerCase().replace(/[\s\u3000，。！？、；：""''（）【】,.!?;:"'()\-–—·]/g, '')
}
export function isNearDuplicate(query, stored) {
  const a = normChars(query)
  const b = normChars(stored)
  const minLen = 12
  if (a.length < minLen || b.length < minLen) return false
  const short = a.length < b.length ? a : b
  const long = a.length < b.length ? b : a
  if (short.length >= minLen && long.includes(short) && short.length / long.length >= 0.6) return true
  const counts = new Map()
  for (const ch of long) counts.set(ch, (counts.get(ch) || 0) + 1)
  let shared = 0
  for (const ch of short) {
    const c = counts.get(ch)
    if (c) { shared++; if (c === 1) counts.delete(ch); else counts.set(ch, c - 1) }
  }
  return (2 * shared) / (short.length + long.length) >= 0.88
}

// ---- deterministic, dependency-free incremental id ----
let seq = 0
function uuid() {
  return 'm' + Date.now().toString(36) + '-' + (++seq).toString(36) + '-' + Math.random().toString(36).slice(2, 8)
}

function nowStamp() {
  const d = new Date()
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

function persist(data) {
  const dir = storeDir()
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(storeFile(), JSON.stringify(data))
}

// Map internal row -> outward-facing result fields used by tool/injection.
function outRow(r) {
  return {
    content: r.text,
    created_at: r.createdAt,
    strength: r.strength,
    memory_type: r.memoryType,
    tags: r.tags,
    group_id: r.groupId,
    chunk_idx: r.idx,
    total_chunks: r.total,
  }
}

export function createStore({ trace = () => {} } = {}) {
  let store
  const embedder = createEmbedder({
    trace,
    onReady: () => { if (store) store.scheduleBackfill() },
  })
  // rows: array of { id, groupId, idx, total, text, createdAt, strength, memoryType, tags, vec? }
  // groups: Map groupId -> { chunks: rows[] } derived on load for stitching.
  let rows = []
  let groups = new Map()

  function load() {
    try {
      const raw = fs.readFileSync(storeFile(), 'utf8')
      const data = JSON.parse(raw)
      rows = Array.isArray(data.rows) ? data.rows : []
    } catch {
      rows = []
    }
    rebuildGroups()
    trace(`store: loaded ${rows.length} chunks from ${storeFile()}`)
  }
  function rebuildGroups() {
    groups = new Map()
    for (const r of rows) {
      if (!groups.has(r.groupId)) groups.set(r.groupId, { chunks: [] })
      groups.get(r.groupId).chunks.push(r)
    }
    for (const g of groups.values()) g.chunks.sort((a, b) => a.idx - b.idx)
  }
  function save() {
    persist({ version: 2, rows })
  }
  function pushRow(row) {
    rows.push(row)
    if (!groups.has(row.groupId)) groups.set(row.groupId, { chunks: [] })
    groups.get(row.groupId).chunks.push(row)
    groups.get(row.groupId).chunks.sort((a, b) => a.idx - b.idx)
  }
  // Rows that were written before the model was ready (no vec) still carry their
  // keywords, so they are always reachable; this backfills their vectors once the
  // model loads, in small batches so the process never stalls.
  let reembedTimer = null
  let reembedRunning = false
  async function backfillEmbeddings() {
    if (reembedRunning) return
    reembedRunning = true
    try {
      while (embedder.isReady()) {
        const pending = rows.filter((r) => !r.vec).slice(0, 16)
        if (pending.length === 0) break
        const ok = await embedRows(pending)
        if (!ok) break
        save()
        await new Promise((r) => setTimeout(r, 0))
      }
    } finally {
      reembedRunning = false
    }
  }
  function scheduleBackfill() {
    if (!embedder.isReady()) return
    if (reembedTimer) clearTimeout(reembedTimer)
    reembedTimer = setTimeout(() => { backfillEmbeddings() }, 200)
  }
  function removeRow(row) {
    rows = rows.filter((r) => r !== row)
    const g = groups.get(row.groupId)
    if (g) {
      g.chunks = g.chunks.filter((r) => r !== row)
      if (g.chunks.length === 0) groups.delete(row.groupId)
    }
  }

  load()
  embedder.warmup()

  async function embedRows(list) {
    const texts = list.map((r) => r.text)
    const vecs = await embedder.embed(texts)
    if (!vecs) return false
    for (let i = 0; i < list.length; i++) list[i].vec = vecs[i]
    return true
  }

  store = {
    embedder,
    count() {
      return rows.length
    },
    isEmbedReady() {
      return embedder.isReady()
    },

    // Add text as one group (chunked at 300/50). Returns number of chunks added.
    async add(text, opts = {}) {
      const clean = String(text || '').trim()
      if (!clean) return 0
      const strength = Math.max(1, Math.min(100, Number.isFinite(opts.strength) ? opts.strength : 50))
      const memoryType = opts.memoryType || opts.type || 'general'
      const rawTags = opts.tags
      const tags = Array.isArray(rawTags)
        ? rawTags.map(String).map((t) => t.trim()).filter(Boolean)
        : typeof rawTags === 'string' && rawTags
          ? rawTags.split(',').map((t) => t.trim()).filter(Boolean)
          : []
      const chunks = chunkText(clean, CHUNK_SIZE, CHUNK_OVERLAP)
      const groupId = uuid()
      const createdAt = nowStamp()
      const created = chunks.map((c, i) => ({
        id: groupId + '_' + i,
        groupId,
        idx: i,
        total: chunks.length,
        text: c,
        createdAt,
        strength,
        memoryType,
        tags,
      }))
      const ok = await embedRows(created) // vec set when model ready
      for (const r of created) pushRow(r)
      save()
      trace(`store: added ${created.length} chunks${ok ? ' (embedded)' : ' (pending-embed keyword-only)'}`)
      return created.length
    },

    async search(query, limit = 5) {
      return store.hybrid(query, { limit })
    },

    // Full hybrid search. Returns rows with {score, source} merged by RRF.
    async hybrid(query, { limit = 5 } = {}) {
      const qtext = String(query || '').trim()
      if (!qtext || rows.length === 0) return []
      const qfeats = tokenize(qtext)
      const candidates = rows.filter((r) => !isNearDuplicate(qtext, r.text))

      // Semantic leg (if any embedded rows and model ready)
      let vecRank = []
      if (embedder.isReady()) {
        const qvec = await embedder.embed([qtext])
        if (qvec && qvec[0]) {
          const withSim = []
          for (const r of candidates) if (r.vec) withSim.push({ row: r, sim: cosine(r.vec, qvec[0]) })
          withSim.sort((a, b) => b.sim - a.sim)
          vecRank = withSim
        }
      }

      // Keyword leg (always)
      const kw = candidates
        .map((r) => ({ row: r, kw: keywordScore(qfeats, tokenize(r.text)) }))
        .filter((x) => x.kw > 0)
        .sort((a, b) => b.kw - a.kw)

      // If no semantic leg, just return keyword results.
      if (vecRank.length === 0) {
        return kw.slice(0, limit).map((x) => ({ ...outRow(x.row), score: x.kw, source: 'keyword' }))
      }
      // Reciprocal-Rank Fusion over both ranked lists.
      const rank = new Map()
      const put = (list) => list.forEach((item, i) => {
        rank.set(item.row.id, (rank.get(item.row.id) || 0) + 1 / (60 + i))
      })
      put(vecRank)
      put(kw)
      const fused = [...rank.entries()].sort((a, b) => b[1] - a[1])
      return fused.slice(0, limit).map(([id, score]) => {
        const row = rows.find((r) => r.id === id)
        return row ? { ...outRow(row), score, source: 'hybrid' } : null
      }).filter(Boolean)
    },

    // mihaji-style recall: relevance top-N + strength²-weighted serendipity,
    // then stitch each hit's group chunk around the hit index.
    async recall(query, { limit = 5 } = {}) {
      const hits = await store.hybrid(query, { limit: limit * 3 + 3 })
      if (hits.length === 0) return []
      const topRelevant = hits.slice(0, limit)
      const remainingPool = hits.slice(limit)
      let chosen = topRelevant.map((h) => h)
      if (remainingPool.length > 0) {
        const picked = weightedSampleNoReplacement(remainingPool, Math.max(0, 2 - topRelevant.length), (h) => h.strength * h.strength)
        chosen = chosen.concat(picked)
      }
      chosen = chosen.slice(0, limit)
      return chosen.map((h) => {
        const g = groups.get(h.groupId)
        const text = g ? stitchChunks(g.chunks.map((c) => c.text), h.idx) : (h.content || '')
        return { ...h, content: text }
      })
    },

    async deleteByQuery(query) {
      const qtext = String(query || '').trim()
      if (!qtext) return 0
      const hits = await store.hybrid(qtext, { limit: 8 })
      const victimIds = new Set()
      for (const h of hits) {
        // Delete whole group when its most relevant chunk matches strongly.
        victimIds.add(h.groupId)
      }
      if (victimIds.size === 0) return 0
      let n = 0
      for (let i = rows.length - 1; i >= 0; i--) {
        if (victimIds.has(rows[i].groupId)) { rows.splice(i, 1); n++ }
      }
      rebuildGroups()
      save()
      return n
    },

    async deleteAll() {
      const n = rows.length
      rows = []
      groups = new Map()
      save()
      return n
    },
  }
  store.scheduleBackfill = scheduleBackfill
  return store
}

// Efraimidis–Spirakis A-Res weighted sampling without replacement.
function weightedSampleNoReplacement(items, n, weightFn) {
  if (n <= 0 || items.length === 0) return []
  const scored = items.map((it) => {
    const w = Math.max(0, weightFn(it))
    return { it, k: w === 0 ? -Infinity : Math.pow(Math.random(), 1 / w) }
  }).filter((x) => x.k !== -Infinity)
  scored.sort((a, b) => b.k - a.k)
  return scored.slice(0, Math.min(n, scored.length)).map((x) => x.it)
}

export { tokenize, chunkText, stitchChunks }
