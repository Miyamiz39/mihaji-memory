// Measure RSS + load time for q8 (quantized int8) embedding model in-process,
// then embed the shared sentence set and dump vectors to q8-vecs.json.
import { pipeline, env } from '@huggingface/transformers'
import { writeFileSync } from 'node:fs'

env.allowRemoteModels = false
env.allowLocalModels = true

const Q8DIR = 'C:/Users/miyaizu/Documents/aFiles/Projects/mihaji-memory/poc-embed/q8model'

const SENTENCES = [
  '今天上海下雨了，出门记得带伞',
  '我家的三花猫叫小花，最爱吃秋刀鱼和芒果',
  '小花最近在减肥，兽医说必须少吃零食',
  '秋刀鱼是日本料理中常见的烤鱼',
  '芒果是热带水果，含丰富的维生素A',
  '这周要发布 dsh-mihaji 插件的新版本',
  '用户的记忆数据库存在 ~/.dsh/mihaji-memory/memory.json',
  'I prefer lightweight architecture without extra services',
  'Memory plugin should embed facts for long-term recall',
  '今天下午三点要和同事开周会',
]

const t0 = Date.now()
// baseline right after module import, before any model bytes are loaded
const baseline = process.memoryUsage().rss / 1048576

const extractor = await pipeline('feature-extraction', Q8DIR, { dtype: 'q8' })
const loadedAt = Date.now()
console.log('q8 load-time(ms) =', loadedAt - t0)
console.log('RSS right-after-load =', (process.memoryUsage().rss / 1048576).toFixed(1), 'MB  (delta +' + ((process.memoryUsage().rss - baseline) / 1048576).toFixed(1) + ' MB)')

const out = await extractor(SENTENCES, { pooling: 'mean', normalize: true })
const dim = out.dims[1]
const n = out.dims[0]
const vecs = []
for (let i = 0; i < n; i++) {
  vecs.push(Array.from(out.data.slice(i * dim, (i + 1) * dim)))
}
const peak = process.memoryUsage().rss / 1048576
console.log('RSS after embed =', peak.toFixed(1), 'MB  (delta vs baseline +' + (peak - baseline).toFixed(1) + ' MB)')
console.log('vec dim =', dim)

writeFileSync(
  'q8-vecs.json',
  JSON.stringify({ sentences: SENTENCES, dims: out.dims, vecs }),
)
console.log('wrote q8-vecs.json')
process.exit(0)
