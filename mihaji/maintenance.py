"""
Mihaji Memory Consolidation — "睡眠巩固" 脚本。

用法:
  python maintenance.py [--dry-run] [--backup-dir ~/memory-backups]

调度 (cron):
  hermes cron create "0 3 * * *" --profile hina \\
    --prompt "直接输出上面记忆维护脚本的执行结果" \\
    --workdir ~/workspace/mihaji_hybrid

功能:
  1. 清理极短噪音 (< 20 chars)
  2. BM25 近似去重 (余弦相似度 > 0.95 的合并)
  3. 导出 JSON 快照备份
  4. 重建 BM25 索引
  5. 报告执行统计
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tags(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def _jaccard_similarity(a: str, b: str) -> float:
    """Quick content similarity via character bigram Jaccard.
    
    Fast, no model needed, good enough for near-duplicate detection.
    """
    if not a or not b:
        return 0.0
    a_bigrams = set(a[i:i+2] for i in range(len(a)-1))
    b_bigrams = set(b[i:i+2] for i in range(len(b)-1))
    if not a_bigrams or not b_bigrams:
        return 0.0
    intersection = a_bigrams & b_bigrams
    union = a_bigrams | b_bigrams
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def consolidate(
    db_path: str,
    dry_run: bool = False,
    backup_dir: str | None = None,
    min_content_length: int = 20,
    dedup_threshold: float = 0.95,
    prune_below_strength: int = 0,
    prune_older_than_days: int = 0,
) -> Dict[str, Any]:
    """Run memory consolidation.

    Returns a report dict with stats.
    """
    try:
        from .store import MihajiStore
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from store import MihajiStore

    store = MihajiStore(db_path=db_path)
    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "db_path": db_path,
        "dry_run": dry_run,
        "before": {},
        "actions": {},
        "after": {},
    }

    # ---- Snapshot before ----
    all_data = store._collection.get()
    docs: List[str] = all_data.get("documents", []) or []
    metas: List[Dict] = all_data.get("metadatas", []) or []
    ids: List[str] = all_data.get("ids", []) or []
    report["before"] = {"total_chunks": len(docs)}

    if not docs:
        report["after"] = {"total_chunks": 0}
        report["summary"] = "数据库为空，无须清理"
        return report

    # ---- Phase 1: 短内容与低价值陈旧记忆清理 ----
    to_delete_set: set = set()
    now = datetime.now()

    for doc, meta, cid in zip(docs, metas, ids):
        # 清理超短噪音
        if len(doc.strip()) < min_content_length:
            to_delete_set.add(cid)
            continue

        # 清理过期低强度记忆
        if prune_below_strength > 0 and prune_older_than_days > 0:
            s = meta.get("strength", 50) if meta else 50
            created_str = meta.get("created_at", "") if meta else ""
            if s < prune_below_strength and created_str:
                try:
                    created_dt = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S")
                    if (now - created_dt).days >= prune_older_than_days:
                        to_delete_set.add(cid)
                except Exception:
                    pass

    report["actions"]["short_noise_removed"] = len(to_delete_set)

    if not dry_run and to_delete_set:
        store._collection.delete(ids=list(to_delete_set))
        logger.info("Deleted %d noise/stale chunks", len(to_delete_set))

    # ---- Phase 2: 近似去重 ----
    # 重新获取或者在内存中过滤已删除项
    if dry_run:
        active_indices = [idx for idx, cid in enumerate(ids) if cid not in to_delete_set]
        active_docs = [docs[i] for i in active_indices]
        active_metas = [metas[i] for i in active_indices]
        active_ids = [ids[i] for i in active_indices]
    else:
        all_data = store._collection.get()
        active_docs = all_data.get("documents", []) or []
        active_metas = all_data.get("metadatas", []) or []
        active_ids = all_data.get("ids", []) or []

    dedup_removed = 0
    dedup_merged: List[str] = []
    processed_ids: set = set()

    # Build group_id lookup
    group_ids = [m.get("group_id", "") if m else "" for m in active_metas]

    # Group by content length bucket
    buckets: Dict[int, List[int]] = {}
    for i, doc in enumerate(active_docs):
        length_bucket = len(doc) // 20
        buckets.setdefault(length_bucket, []).append(i)

    # Check same bucket and adjacent bucket (bucket + 1)
    bucket_keys = sorted(buckets.keys())
    for b_idx, b_key in enumerate(bucket_keys):
        candidates = list(buckets[b_key])
        if b_idx + 1 < len(bucket_keys) and bucket_keys[b_idx + 1] == b_key + 1:
            next_candidates = buckets[bucket_keys[b_idx + 1]]
        else:
            next_candidates = []

        for i in candidates:
            if active_ids[i] in processed_ids:
                continue

            # Compare within current bucket
            for j in candidates:
                if i >= j or active_ids[j] in processed_ids:
                    continue
                if group_ids[i] and group_ids[i] == group_ids[j]:
                    continue
                sim = _jaccard_similarity(active_docs[i], active_docs[j])
                if sim >= dedup_threshold:
                    s_i = active_metas[i].get("strength", 50) if active_metas[i] else 50
                    s_j = active_metas[j].get("strength", 50) if active_metas[j] else 50
                    if s_i >= s_j:
                        to_remove = j
                    else:
                        to_remove = i
                    remove_id = active_ids[to_remove]
                    if remove_id not in processed_ids:
                        dedup_merged.append(remove_id)
                        processed_ids.add(remove_id)
                        dedup_removed += 1
                    if to_remove == i:
                        break

            if active_ids[i] in processed_ids:
                continue

            # Compare with adjacent bucket
            for j in next_candidates:
                if active_ids[j] in processed_ids:
                    continue
                if group_ids[i] and group_ids[i] == group_ids[j]:
                    continue
                sim = _jaccard_similarity(active_docs[i], active_docs[j])
                if sim >= dedup_threshold:
                    s_i = active_metas[i].get("strength", 50) if active_metas[i] else 50
                    s_j = active_metas[j].get("strength", 50) if active_metas[j] else 50
                    if s_i >= s_j:
                        to_remove = j
                    else:
                        to_remove = i
                    remove_id = active_ids[to_remove]
                    if remove_id not in processed_ids:
                        dedup_merged.append(remove_id)
                        processed_ids.add(remove_id)
                        dedup_removed += 1
                    if to_remove == i:
                        break

    report["actions"]["dedup_removed"] = dedup_removed

    if not dry_run and dedup_merged:
        store._collection.delete(ids=dedup_merged)
        logger.info("Merged %d near-duplicate chunks", dedup_removed)

    # ---- Phase 3: 备份 ----
    if backup_dir and not dry_run:
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        all_data = store._collection.get()
        backup_file = backup_path / f"mihaji_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        report["actions"]["backup_path"] = str(backup_file)
        logger.info("Backup saved: %s", backup_file)

    # ---- Snapshot after ----
    if dry_run:
        after_count = len(docs) - len(to_delete_set) - dedup_removed
        report["after"] = {"total_chunks": max(0, after_count)}
    else:
        final = store._collection.get()
        final_docs = final.get("documents", []) or []
        report["after"] = {"total_chunks": len(final_docs)}

    # ---- Phase 4: BM25 reindex ----
    if not dry_run:
        store._mark_bm25_stale()
        report["actions"]["bm25_reindex"] = "pending (auto on next query)"

    report["summary"] = (
        f"{'[DRY RUN] ' if dry_run else ''}"
        f"清理前 {report['before']['total_chunks']} 条 → "
        f"清理后 {report['after']['total_chunks']} 条 "
        f"(去噪音/过期 {report['actions']['short_noise_removed']}, "
        f"去重 {report['actions']['dedup_removed']})"
    )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mihaji Memory Consolidation")
    parser.add_argument("--db-path", default=None, help="ChromaDB path (default: ~/.hermes/profiles/hina/mihaji_memory)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no changes")
    parser.add_argument("--backup-dir", default=None, help="Directory for JSON backup (default: no backup)")
    parser.add_argument("--min-length", type=int, default=20, help="Minimum content length (default: 20)")
    parser.add_argument("--dedup-threshold", type=float, default=0.95, help="Jaccard similarity threshold (default: 0.95)")
    parser.add_argument("--prune-below-strength", type=int, default=0, help="Prune memories with strength below this (requires --prune-older-than-days)")
    parser.add_argument("--prune-older-than-days", type=int, default=0, help="Prune memories older than N days if strength is low")
    args = parser.parse_args()

    db_path = args.db_path or os.path.expanduser("~/.hermes/profiles/hina/mihaji_memory")

    print(f"🧹 Mihaji 记忆巩固 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'⚠️ DRY RUN - 不会修改数据' if args.dry_run else '✅ 将执行实际清理'}")
    print()

    report = consolidate(
        db_path=db_path,
        dry_run=args.dry_run,
        backup_dir=args.backup_dir,
        min_content_length=args.min_length,
        dedup_threshold=args.dedup_threshold,
        prune_below_strength=args.prune_below_strength,
        prune_older_than_days=args.prune_older_than_days,
    )

    print(f"📊 统计报告")
    print(f"  清理前: {report['before']['total_chunks']} 条")
    print(f"  短噪音/过期: {report['actions']['short_noise_removed']} 条")
    print(f"  去重:   {report['actions']['dedup_removed']} 条")
    if "backup_path" in report["actions"]:
        print(f"  备份:   {report['actions']['backup_path']}")
    print(f"  清理后: {report['after']['total_chunks']} 条")
    print()
    print(report["summary"])


if __name__ == "__main__":
    main()
