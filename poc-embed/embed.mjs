import { pipeline, env } from '@huggingface/transformers'

// Force fully offline: never fetch from HF hub.
env.allowRemoteModels = false
env.allowLocalModels = true

const SNAPSHOT =
  'C:/Users/miyaizu/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/main'

async function embed(extractor, texts) {
  const out = await extractor(texts, { pooling: 'mean', normalize: true })
  // out is a Tensor with shape [n, dim]
  const dim = out.dims[1]
  const data = out.data
  const vecs = []
  const n = out.dims[0]
  for (let i = 0; i < n; i++) {
    vecs.push(Array.from(data.slice(i * dim, (i + 1) * dim)))
  }
  return { dims: out.dims, vecs }
}

function cosine(a, b) {
  let dot = 0
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i]
  return dot
}

const extractor = await pipeline('feature-extraction', SNAPSHOT)
console.log('model loaded OK')

const texts = [
  '我养了一只三花猫叫小花，它特别喜欢芒果和秋刀鱼',
  '我的猫最近在减肥，兽医说要少吃零食',
  '今天上海的天气怎么样？',
  '芒果是热带水果，营养丰富',
  'I bought a new laptop yesterday',
]

const { vecs } = await embed(extractor, texts)
console.log('vec dim =', vecs[0].length)

const names = ['小花猫爱吃芒果', '猫减肥兽医', '上海天气', '芒果营养', 'laptop英文']
for (let i = 0; i < names.length; i++) {
  const row = names.map((_, j) => cosine(vecs[i], vecs[j]).toFixed(2))
  console.log(names[i], '=>', row.join('  '))
}
