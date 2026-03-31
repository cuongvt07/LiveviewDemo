import { promises as fs } from 'node:fs';
import path from 'node:path';

const SOURCE_ROOT = path.resolve(process.cwd(), 'src');
const CHECK_EXTENSIONS = new Set(['.ts', '.tsx', '.js', '.jsx', '.css', '.json', '.html']);
// Match common mojibake sequences without flagging valid Vietnamese diacritics.
const MOJIBAKE_PATTERN = /(Ã|Â|Ä|â†|âœ|â€“|â€”|â€|ðŸ|�)/;

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await walk(fullPath));
      continue;
    }
    if (CHECK_EXTENSIONS.has(path.extname(entry.name))) {
      files.push(fullPath);
    }
  }
  return files;
}

function findMojibakeLines(content) {
  const matches = [];
  const lines = content.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    if (MOJIBAKE_PATTERN.test(lines[i])) {
      matches.push({ line: i + 1, text: lines[i].trim() });
    }
  }
  return matches;
}

async function main() {
  const files = await walk(SOURCE_ROOT);
  const violations = [];

  for (const file of files) {
    const content = await fs.readFile(file, 'utf8');
    const matches = findMojibakeLines(content);
    if (matches.length > 0) {
      violations.push({ file, matches });
    }
  }

  if (violations.length === 0) {
    console.log('Encoding check passed: no mojibake patterns found in src.');
    return;
  }

  console.error('Encoding check failed: suspicious mojibake text found.');
  for (const item of violations) {
    for (const match of item.matches) {
      console.error(`- ${path.relative(process.cwd(), item.file)}:${match.line}: ${match.text}`);
    }
  }
  process.exitCode = 1;
}

main().catch((error) => {
  console.error('Encoding check crashed:', error);
  process.exitCode = 1;
});
