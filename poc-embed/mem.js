// Measure RSS + load time for fp32 embedding model in-process.
import { createEmbedder } from '../dsh-mihaji/lib/embed.js'

const t0 = Date.now()
const trace = (l) => console.log('[trace]', l)
const e = createEmbedder({ trace })
e.warmup()
const baseline = process.memoryUsage().rss / 1048576
await new Promise((r) => setTimeout(r, 100))
// wait until ready or timeout 15s
let ready = false
for (let i = 0; i < 30; i++) {
  if (e.isReady()) { ready = true; break }
  await new Promise((r) => setTimeout(r, 500))
}
const loadedAt = Date.now()
console.log('ready =', ready, 'load-time(ms) =', loadedAt - t0)
if (ready) {
  await e.embed(['今天上海下雨', '我家的猫叫小花'])
}
const peak = process.memoryUsage().rss / 1048576
console.log('RSS baseline(ready+2 embeds) =', peak.toFixed(1), 'MB  (delta vs empty = +' + (peak - baseline).toFixed(1) + ' MB)')
process.exit(0)
