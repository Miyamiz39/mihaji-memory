import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const memoryPath = path.join(os.homedir(), '.dsh', 'mihaji-memory', 'memory.json');

if (!fs.existsSync(memoryPath)) {
  console.error(`Memory file not found: ${memoryPath}`);
  process.exit(1);
}

// 1. Backup first
const backupPath = path.join(
  path.dirname(memoryPath),
  `memory.backup-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
);
console.log(`Backing up to: ${backupPath}`);
fs.copyFileSync(memoryPath, backupPath);

// 2. Read and filter
const raw = fs.readFileSync(memoryPath, 'utf8');
const data = JSON.parse(raw);

const beforeCount = data.rows ? data.rows.length : 0;
const filteredRows = (data.rows || []).filter(row => {
  const strength = Number(row.strength ?? 0);
  return strength > 0;
});
const afterCount = filteredRows.length;
const removedCount = beforeCount - afterCount;

data.rows = filteredRows;

// 3. Write back atomically
const tempPath = `${memoryPath}.tmp`;
fs.writeFileSync(tempPath, JSON.stringify(data, null, 2), 'utf8');
fs.renameSync(tempPath, memoryPath);

console.log(`Done! Removed ${removedCount} entries with strength == 0.`);
console.log(`Remaining: ${afterCount} entries.`);
