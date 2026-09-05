// Load fp32 model and dump vectors for the shared sentence set to fp32-vecs.json.
import { pipeline, env } from '@huggingface/transformers'
import { writeFileSync } from 'node:fs'

env.allowRemoteModels = false
env.allowLocalModels = true

const SNAPSHOT =
  'C:/Users/miyaizu/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/main'

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
const extractor = await pipeline('feature-extraction', SNAPSHOT)
console.log('fp32 load-time(ms) =', Date.now() - t0)

const out = await extractor(SENTENCES, { pooling: 'mean', normalize: true })
const dim = out.dims[1]
const n = out.dims[0]
const vecs = []
for (let i = 0; i < n; i++) {
  vecs.push(Array.from(out.data.slice(i * dim, (i + 1) * dim)))
}
writeFileSync(
  'fp32-vecs.json',
  JSON.stringify({ sentences: SENTENCES, dims: out.dims, vecs }),
)
console.log('wrote fp32-vecs.json, dim =', dim)
process.exit(0)
