/**
 * Mihaji 记忆库一键清理与压缩脚本
 * 
 * 功能：
 * 1. 自动备份当前的 memory.json（带时间戳）
 * 2. 彻底清除 strength <= 0 的失效废弃记忆（1500+ 条死重）
 * 3. 统计并输出清理前后的大小与条目对比
 * 4. 安全写入新文件
 * 
 * 运行方式：
 *   node clean-memory.mjs
 */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const memoryDir = path.join(os.homedir(), '.dsh', 'mihaji-memory');
const memoryFile = path.join(memoryDir, 'memory.json');

if (!fs.existsSync(memoryFile)) {
  console.error(`❌ 未找到记忆库文件: ${memoryFile}`);
  process.exit(1);
}

console.log('🐾 [Mihaji Memory Cleaner] 正在读取记忆库...');
const raw = fs.readFileSync(memoryFile, 'utf8');
const beforeBytes = Buffer.byteLength(raw, 'utf8');
let data;

try {
  data = JSON.parse(raw);
} catch (e) {
  console.error('❌ 解析 memory.json 失败:', e.message);
  process.exit(1);
}

const rows = Array.isArray(data.rows) ? data.rows : [];
const initialCount = rows.length;

// 1. 创建备份
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const backupFile = path.join(memoryDir, `memory.backup-${timestamp}.json`);
fs.writeFileSync(backupFile, raw, 'utf8');
console.log(`📦 已创建安全备份: ${path.basename(backupFile)}`);

// 2. 过滤掉 strength <= 0 的无用记忆
const validRows = rows.filter((r) => {
  const s = Number(r.strength);
  return !Number.isNaN(s) && s > 0;
});

const prunedCount = initialCount - validRows.length;
data.rows = validRows;

// 3. 写入新数据
const formatted = JSON.stringify(data, null, 2);
fs.writeFileSync(memoryFile, formatted, 'utf8');
const afterBytes = Buffer.byteLength(formatted, 'utf8');

console.log('\n✨ 清理完成！数据统计：');
console.log(`- 清理前总条目: ${initialCount} 条 (${(beforeBytes / 1024 / 1024).toFixed(2)} MB)`);
console.log(`- 移除失效记忆 (strength <= 0): ${prunedCount} 条`);
console.log(`- 保留有效记忆: ${validRows.length} 条 (${(afterBytes / 1024 / 1024).toFixed(2)} MB)`);
console.log(`- 瘦身幅度: -${((beforeBytes - afterBytes) / 1024 / 1024).toFixed(2)} MB (${Math.round((1 - afterBytes / beforeBytes) * 100)}%)`);
console.log('\n💡 提示：若 DSH 正在后台运行，可以在托盘里点一下【重启 DSH】重新载入干净的记忆库~ 🐾');
