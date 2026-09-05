// TEMP sanity test for lib/sessionsearch.js with a mocked sessionQuery.
import { createSessionSearch } from '../dsh-mihaji/lib/sessionsearch.js'

const mkUser = (seq, time, text) => ({ type: 'user/message', seq, time, data: { id: `u${seq}`, role: 'user', content: [{ type: 'text', text }], source: { kind: 'user' } } })
const mkAsst = (seq, time, text) => ({ type: 'assistant/message', seq, time, data: { turn: 1, step: 1, message: { id: `a${seq}`, role: 'assistant', content: [{ type: 'text', text }], source: { kind: 'model', provider: 'x', model: 'y' } } } })
const mkTool = (seq, time) => ({ type: 'tool/call', seq, time, data: { turn: 1, step: 1, callId: `c${seq}`, name: 'some_tool', arguments: '{}' } })

const SESSIONS = {
  s1: { header: { id: 's1', createdAt: 2000000, cwd: 'C:/w' }, events: [mkUser(1, 1000, '今天聊了芒果和秋刀鱼'), mkAsst(2, 2000, '好的我记住了'), mkTool(3, 3000)] },
  s2: { header: { id: 's2', createdAt: 1000000, cwd: 'C:/w', origin: 'subagent' }, events: [mkUser(1, 1000, '内部子代理：芒果')] },
  s3: { header: { id: 's3', createdAt: 3000000, cwd: 'C:/x' }, events: [mkUser(1, 500, '上海下雨记得带伞'), mkAsst(2, 600, '嗯嗯'), mkUser(3, 700, '芒果好甜'), mkAsst(4, 800, '是的呀')] },
}
const sq = {
  async listSessions() {
    return Object.values(SESSIONS).map((s) => ({ header: s.header, live: false, persisted: true }))
  },
  async readSession(id) {
    const s = SESSIONS[id]
    if (!s) throw new Error('no session ' + id)
    return { session: s.header, events: s.events }
  },
  async readEvent({ sessionId, seq, before, after }) {
    const s = SESSIONS[sessionId]
    const idx = s.events.findIndex((e) => e.seq === seq)
    const from = Math.max(0, idx - before)
    const to = Math.min(s.events.length, idx + after + 1)
    return { events: s.events.slice(from, to) }
  },
  async readTitle(id) {
    return SESSIONS[id] ? { title: '标题-' + id } : undefined
  },
}

const ss = createSessionSearch({ sq, trace: (l) => console.log('[trace]', l) })
const show = (label, o) => console.log(`\n=== ${label} ===\n` + JSON.stringify(o, null, 2))

// browse (subagent s2 excluded)
show('browse', await ss.run({}))

// discover query=芒果 (should rank s3 above s1; subagent excluded)
show('discover', await ss.run({ query: '芒果' }))

// discover sort oldest detail full
show('discover-full', await ss.run({ query: '芒果', sort: 'oldest', detail: 'full', limit: 5 }))

// read whole s3
show('read', await ss.run({ session_id: 's3' }))

// scroll s3 around seq 3
show('scroll', await ss.run({ session_id: 's3', around_seq: 3, window: 1 }))

// error: no such session
show('read-missing', await ss.run({ session_id: 'nope' }))
