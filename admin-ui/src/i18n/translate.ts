import translations from './translations.json';

const DEFAULT_LOCALE = 'vi';

export function t(key: string, vars?: Record<string, any>, locale = DEFAULT_LOCALE): string {
  const parts = key.split('.');
  let node: any = (translations as any)[locale] || {};
  for (const p of parts) {
    if (node && typeof node === 'object' && p in node) node = node[p];
    else { node = null; break; }
  }
  let str = (typeof node === 'string') ? node : key;
  if (vars) {
    for (const k of Object.keys(vars)) {
      str = str.replace(new RegExp(`\\{${k}\\}`, 'g'), String(vars[k]));
    }
  }
  return str;
}

export default t;
