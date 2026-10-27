import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const root = fileURLToPath(new URL('../../', import.meta.url));
const types = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.png': 'image/png' };
const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/') { res.writeHead(302, { Location: '/frontend/studio/' }); res.end(); return; }
    let pathname = decodeURIComponent(url.pathname); if (pathname.endsWith('/')) pathname += 'index.html';
    const file = path.resolve(root, '.' + pathname);
    const studioRoot = path.join(root, 'frontend', 'studio') + path.sep;
    const image = path.join(root, 'docs', 'character-design', 'kyoko-design-ver1.png');
    const logo = path.join(root, 'docs', 'character-design', 'logo', 'kyoko-logo.png');
    if (!file.startsWith(studioRoot) && file !== image && file !== logo) { res.writeHead(403); res.end('アクセスできません'); return; }
    const body = await readFile(file); res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache' }); res.end(body);
  } catch { res.writeHead(404); res.end('見つかりません'); }
});
server.listen(Number(process.env.PORT || 4173), '127.0.0.1', () => console.log(`京子の収録室: http://localhost:${server.address().port}/frontend/studio/`));
