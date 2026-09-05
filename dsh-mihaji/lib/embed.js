// dsh-mihaji — sentence embedding via local ONNX (transformers.js, fully
// offline). Loads the same MiniLM multilingual model Hermes caches locally, so
// migrated Hermes memories are comparable with newly embedded text.
//
// Model variant: INT8-quantized (q8) by default. Measured vs fp32 on this
// machine: RSS ~450 MB vs ~770 MB in-process (~330 MB saved), per-sentence
// cosine agreement with fp32 ≥ 0.992, top-3 rank preservation 97% — a safe
// swap. transformers.js dtype 'q8' expects `onnx/model_quantized.onnx`, which
// the HF cache does not ship under that name, so on first use we copy the
// AVX2-quantized file from the Hermes HF cache snapshot into a managed dir
// under $DSH_HOME/mihaji-memory/models/q8 (one time, async). If no quantized
// file exists or the q8 session fails to load, we fall back to fp32.
//
// Key design: lazy + non-blocking. Loading MiniLM into ONNX Runtime takes
// ~1-2s and the one-time copy ~113MB, but both run inside a background
// warmup — we never block the agent loop on them. The store treats embedding
// as optional: if the model is not ready yet, retrieval degrades to pure
// keyword (mirrors mihaji's cold-start BM25-only behaviour).
//
// Model discovery order (all resolved offline; never hits the HF hub):
//   1. process.env.DSH_MIHAJI_MODEL → a directory holding onnx/model_quantized.onnx (q8)
//      or onnx/model.onnx (fp32) — explicit override wins
//   2. managed q8 dir under $DSH_HOME/mihaji-memory/models/q8 (already provisioned)
//   3. any HF-cache snapshot carrying onnx/model_quint8_avx2.onnx → provision it
//      into the managed dir, then use q8
//   4. any HF-cache snapshot carrying onnx/model.onnx → fp32
//   5. none found → embedding disabled (keyword-only mode)

import path from 'node:path'
import os from 'node:os'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import { pipeline, env } from '@huggingface/transformers'

// Fully offline: refuse remote fetches so the plugin never stalls on the network.
env.allowRemoteModels = false
env.allowLocalModels = true

const MODEL_REPO = 'models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2'
// Quantized ONNX shipped in the HF cache; validated to load under onnxruntime-node.
const Q8_QUANT_FILE = 'model_quint8_avx2.onnx'
// Filename transformers.js dtype 'q8' resolves (DEFAULT_DTYPE_SUFFIX_MAPPING.q8).
const Q8_MODEL_FILE = 'model_quantized.onnx'
// Small files that must sit next to onnx/ so transformers.js can build the
// tokenizer + model config. Only the ones present in the snapshot are copied.
const Q8_SIDE_FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'config_sentence_transformers.json',
  'modules.json',
  'sentence_bert_config.json',
]

function dshHome() {
  return process.env.DSH_HOME || path.join(os.homedir(), '.dsh')
}
function managedQ8Dir() {
  return path.join(dshHome(), 'mihaji-memory', 'models', 'q8')
}
function hfSnapshotDirs() {
  const base = path.join(os.homedir(), '.cache', 'huggingface', 'hub', MODEL_REPO, 'snapshots')
  if (!fs.existsSync(base)) return []
  return ['main', ...fs.readdirSync(base).filter((n) => n !== 'main')].map((n) => path.join(base, n))
}

// Copy the quantized model + side files from an HF-cache snapshot into the
// managed dir (idempotent per snapshot; partial failures roll the dir back so
// the next boot retries cleanly). Async so the 113MB copy never blocks the loop.
async function provisionQ8(snapshotDir) {
  const target = managedQ8Dir()
  const tmp = path.join(target, 'onnx', `.tmp-${process.pid}-${Date.now()}`)
  try {
    await fsp.mkdir(path.join(target, 'onnx'), { recursive: true })
    for (const f of Q8_SIDE_FILES) {
      const src = path.join(snapshotDir, f)
      if (fs.existsSync(src)) await fsp.copyFile(src, path.join(target, f))
    }
    await fsp.copyFile(path.join(snapshotDir, 'onnx', Q8_QUANT_FILE), tmp)
    await fsp.rename(tmp, path.join(target, 'onnx', Q8_MODEL_FILE))
    return fs.existsSync(path.join(target, 'onnx', Q8_MODEL_FILE))
  } catch (e) {
    try { await fsp.rm(target, { recursive: true, force: true }) } catch { /* ignore */ }
    return false
  }
}

async function planModel() {
  const explicit = process.env.DSH_MIHAJI_MODEL
  if (explicit) {
    if (fs.existsSync(path.join(explicit, 'onnx', Q8_MODEL_FILE))) return { dir: explicit, dtype: 'q8' }
    if (fs.existsSync(path.join(explicit, 'onnx', 'model.onnx'))) return { dir: explicit, dtype: 'fp32' }
  }
  const managed = managedQ8Dir()
  if (fs.existsSync(path.join(managed, 'onnx', Q8_MODEL_FILE))) return { dir: managed, dtype: 'q8' }
  for (const snap of hfSnapshotDirs()) {
    if (fs.existsSync(path.join(snap, 'onnx', Q8_QUANT_FILE))) {
      if (await provisionQ8(snap)) return { dir: managed, dtype: 'q8' }
    }
  }
  for (const snap of hfSnapshotDirs()) {
    if (fs.existsSync(path.join(snap, 'onnx', 'model.onnx'))) return { dir: snap, dtype: 'fp32' }
  }
  return null
}

// Synchronous fp32 fallback source (used when a q8 session fails at load time).
function fp32Snapshot() {
  for (const snap of hfSnapshotDirs()) {
    if (fs.existsSync(path.join(snap, 'onnx', 'model.onnx'))) return snap
  }
  return null
}

export function createEmbedder({ trace = () => {}, onReady } = {}) {
  let extractorPromise = null
  let ready = false
  let planned = false
  let plan = null // { dir, dtype } | null (null = nothing usable found)

  async function ensureLoaded() {
    if (extractorPromise) return extractorPromise
    if (!planned) {
      planned = true
      plan = await planModel()
      if (!plan) {
        trace('embed: no local model found — keyword-only mode')
        return null
      }
      trace(`embed: using ${plan.dtype} model from ${plan.dir}`)
    }
    const opts = plan.dtype === 'q8' ? { dtype: 'q8' } : undefined
    extractorPromise = pipeline('feature-extraction', plan.dir, opts).then((ext) => {
      ready = true
      trace(`embed: model ready (${plan.dtype})`)
      if (typeof onReady === 'function') {
        try { onReady() } catch { /* ignore */ }
      }
      return ext
    }).catch((e) => {
      extractorPromise = null
      ready = false
      // q8 failed to load (e.g. runtime rejects the quantized graph on this CPU) —
      // degrade to fp32 once instead of dropping embeddings entirely.
      if (plan && plan.dtype === 'q8') {
        const snap = fp32Snapshot()
        if (snap) {
          plan = { dir: snap, dtype: 'fp32' }
          trace(`embed: q8 load failed (${e && e.message ? e.message : e}), falling back to fp32`)
          return null
        }
      }
      trace(`embed: load FAILED (keyword-only): ${e && e.message ? e.message : String(e)}`)
      return null
    })
    return extractorPromise
  }

  return {
    isReady() {
      return ready
    },
    modelDir() {
      return plan ? plan.dir : null
    },
    // Kick off loading in the background; never await it in the hot path.
    warmup() {
      ensureLoaded()
    },
    async embed(texts) {
      if (!Array.isArray(texts) || texts.length === 0) return null
      const ext = await ensureLoaded()
      if (!ext) return null
      try {
        const out = await ext(texts, { pooling: 'mean', normalize: true })
        const dim = out.dims[1]
        const data = out.data
        const vecs = []
        const n = out.dims[0]
        for (let i = 0; i < n; i++) vecs.push(Array.from(data.slice(i * dim, (i + 1) * dim)))
        return vecs
      } catch (e) {
        trace(`embed: run failed: ${e && e.message ? e.message : String(e)}`)
        return null
      }
    },
  }
}

export function cosine(a, b) {
  let dot = 0
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i]
  return dot
}
