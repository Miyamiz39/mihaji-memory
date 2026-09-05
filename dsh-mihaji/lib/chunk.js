// dsh-mihaji — text chunking + stitched context, ported from Hermes
// mihaji/store.py (CHUNK_SIZE=300, CHUNK_OVERLAP=50). Recall returns the chunk
// plus ±1 neighbours stitched together so sentences are never cut mid-thought.

export const CHUNK_SIZE = 300
export const CHUNK_OVERLAP = 50

export function chunkText(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  if (!text) return []
  if (size <= overlap || size <= 0) return [text]
  if (text.length <= size) return [text]
  const chunks = []
  let start = 0
  const step = size - overlap
  while (start < text.length) {
    const end = start + size
    const chunk = text.slice(start, end)
    if (chunk) chunks.push(chunk)
    start += step
    if (end >= text.length) break
  }
  return chunks
}

// Merge a hit chunk with its immediate neighbours into seamless prose.
export function stitchChunks(groupChunks, idx) {
  if (!groupChunks || groupChunks.length === 0) return ''
  if (groupChunks.length === 1) return groupChunks[0]
  const start = Math.max(0, idx - 1)
  const end = Math.min(groupChunks.length - 1, idx + 1)
  const parts = []
  for (let i = start; i <= end; i++) {
    const doc = groupChunks[i]
    if (!doc) continue
    if (parts.length === 0) {
      parts.push(doc)
      continue
    }
    const prev = parts[parts.length - 1]
    let merged = false
    // Try to fuse the overlap boundary (prefer larger common suffix/prefix).
    for (let ol = CHUNK_OVERLAP + 15; ol > Math.max(0, CHUNK_OVERLAP - 20); ol--) {
      if (ol < prev.length && ol < doc.length && prev.slice(-ol) === doc.slice(0, ol)) {
        parts.push(doc.slice(ol))
        merged = true
        break
      }
    }
    if (!merged) parts.push(' ' + doc)
  }
  return parts.join('')
}
