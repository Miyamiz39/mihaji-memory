// Same harness/structure as mem-q8.js but fp32 model, for a fair RSS comparison.
import { pipeline, env } from '@huggingface/transformers'

env.allowRemoteModels = false
env.allowLocalModels = true

const SNAPSHOT =
  'C:/Users/miyaizu/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/main'

const SENTENCES = [
  '今天上海下雨了，出门记得带伞',
  '我家的三花猫叫小花，最爱吃秋刀鱼和芒果',
  '小花最近在减肥，兽医说必须少吃零食',
]

const t0 = Date.now()
const baseline = process.memoryUsage().rss / 1048576
console.log('baseline (after import) =', baseline.toFixed(1), 'MB')

const extractor = await pipeline('feature-extraction', SNAPSHOT)
const after = process.memoryUsage().rss / 1048576
console.log('fp32 load-time(ms) =', Date.now() - t0)
console.log('RSS right-after-load =', after.toFixed(1), 'MB  (delta +' + (after - baseline).toFixed(1) + ' MB)')

await extractor(SENTENCES, { pooling: 'mean', normalize: true })
const peak = process.memoryUsage().rss / 1048576
console.log('RSS after embed =', peak.toFixed(1), 'MB  (delta vs baseline +' + (peak - baseline).toFixed(1) + ' MB)')
process.exit(0)
