// Compare fp32 vs q8 embeddings of the same sentences:
// 1) model agreement: per-sentence cosine(fp32_vec, q8_vec)
// 2) rank preservation: for each sentence, does q8 keep the same top-3 nearest
//    neighbours (by cosine) as fp32?
import { readFileSync } from 'node:fs'

const fp = JSON.parse(readFileSync('fp32-vecs.json', 'utf8'))
const q8 = JSON.parse(readFileSync('q8-vecs.json', 'utf8'))

function cosine(a, b) {
  let dot = 0
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i]
  return dot
}

console.log('== per-sentence fp32-vs-q8 agreement ==')
let sum = 0
let min = 1
for (let i = 0; i < fp.vecs.length; i++) {
  const c = cosine(fp.vecs[i], q8.vecs[i])
  sum += c
  if (c < min) min = c
  console.log((c < 0.999 ? ' ' : '') + c.toFixed(5), '|', fp.sentences[i])
}
console.log('mean cos =', (sum / fp.vecs.length).toFixed(5), ' min cos =', min.toFixed(5))

console.log('\n== top-3 nearest rank preservation (per row) ==')
function topK(i, vecs, k) {
  const sims = vecs.map((v, j) => ({ j, s: cosine(vecs[i], v) }))
  sims.sort((a, b) => b.s - a.s)
  return sims.filter((x) => x.j !== i).slice(0, k).map((x) => x.j).sort((a, b) => a - b)
}
let hit = 0
let total = 0
for (let i = 0; i < fp.vecs.length; i++) {
  const a = topK(i, fp.vecs, 3)
  const b = topK(i, q8.vecs, 3)
  const overlap = a.filter((x) => b.includes(x)).length
  hit += overlap
  total += 3
  const mark = overlap === 3 ? 'OK ' : overlap === 2 ? '~  ' : 'BAD'
  console.log(mark, 'fp32 top3:', a.map((j) => j).join(','), ' q8 top3:', b.map((j) => j).join(','), '|', fp.sentences[i])
}
console.log('\nrank-preservation =', hit + '/' + total, '=' + ((100 * hit) / total).toFixed(0) + '%')

// global mean abs difference of full pairwise matrix
let dsum = 0
let dcnt = 0
let maxd = 0
for (let i = 0; i < fp.vecs.length; i++) {
  for (let j = i + 1; j < fp.vecs.length; j++) {
    const d = Math.abs(cosine(fp.vecs[i], fp.vecs[j]) - cosine(q8.vecs[i], q8.vecs[j]))
    dsum += d
    dcnt++
    if (d > maxd) maxd = d
  }
}
console.log('\npairwise-cosine |fp32-q8| diff: mean =', (dsum / dcnt).toFixed(5), ' max =', maxd.toFixed(5))
