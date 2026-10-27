// 京子の告知映像用。背景・字幕は同じCanvasから録画とPNGに出力する。
const navy = '#263951';
function box(c, x, y, w, h, r, color) {
  c.fillStyle = color; c.beginPath(); c.roundRect(x, y, w, h, r); c.fill();
}
export function background(c, w, h, kind) {
  const night = kind === 'night', paper = kind === 'paper';
  const g = c.createLinearGradient(0, 0, w, h);
  g.addColorStop(0, night ? '#1c2b43' : paper ? '#fffaf2' : '#f5f7fb');
  g.addColorStop(1, night ? '#46566e' : paper ? '#eee5db' : '#dde5ef');
  c.fillStyle = g; c.fillRect(0, 0, w, h);
  c.strokeStyle = night ? '#ffffff12' : '#2639510c'; c.lineWidth = 1;
  const step = w / 12;
  for (let x = step / 2; x < w; x += step) {
    for (let y = step / 2; y < h; y += step) {
      c.beginPath(); c.arc(x, y, 1.5, 0, Math.PI * 2); c.stroke();
    }
  }
  // 装飾は端に寄せ、人物と字幕の後ろを静かに保つ。
  c.strokeStyle = night ? '#dae4f04d' : '#52698630'; c.lineWidth = w * .002;
  for (const r of [.12, .16, .20]) {
    c.beginPath(); c.arc(w * .94, h * .14, w * r, 0, Math.PI * 2); c.stroke();
  }
  c.fillStyle = night ? '#fff9e9' : '#c2cedd';
  c.beginPath(); c.arc(w * .88, h * .105, w * .025, 0, Math.PI * 2); c.fill();
  box(c, w * .045, h * .91, w * .91, h * .003, 2, night ? '#ffffff40' : '#26395130');
}
export function caption(c, w, h, text, style) {
  if (!text) return;
  let size = w > h ? 46 : 50, lines;
  // 長文も欠落させない。極端な長文は台本側で分割する。
  do {
    c.font = `600 ${size}px "Yu Gothic", "Meiryo", sans-serif`;
    lines = []; let row = '';
    for (const ch of text) {
      if (c.measureText(row + ch).width > w * .78 && row) { lines.push(row); row = ''; }
      row += ch;
    }
    if (row) lines.push(row);
    if (lines.length * size * 1.55 <= h * .28 || size <= 24) break;
    size -= 2;
  } while (true);
  const height = lines.length * size * 1.55 + size * .85;
  const y = h * .90 - height;
  box(c, w * .07, y, w * .86, height, 22, style === 'light' ? '#fffffff2' : '#263951f2');
  box(c, w * .07, y + 22, w * .006, Math.max(12, height - 44), 3, '#b9cbe0');
  c.fillStyle = style === 'light' ? navy : '#ffffff'; c.textAlign = 'center';
  lines.forEach((row, i) => c.fillText(row, w / 2, y + size * (1.35 + i * 1.55)));
  c.textAlign = 'left';
}
export function overlay(c, w, h, title, days, logo) {
  const f = w > h ? 32 : 34;
  if (logo?.complete && logo.naturalWidth) {
    const lw = w * .23, lh = lw * logo.naturalHeight / logo.naturalWidth;
    c.drawImage(logo, w * .05, h * .035, lw, lh);
  }
  if (title) {
    c.font = `600 ${f}px "Yu Gothic", sans-serif`;
    const bw = Math.min(w * .8, c.measureText(title).width + 52);
    box(c, w * .05, h * .20, bw, f * 1.9, 16, '#fffffff0');
    c.save(); c.beginPath(); c.rect(w * .05, h * .20, bw, f * 1.9); c.clip();
    c.fillStyle = navy; c.fillText(title, w * .05 + 26, h * .20 + f * 1.28, bw - 40); c.restore();
  }
  if (days !== '') {
    const label = Number(days) === 0 ? '本日デビュー' : `あと${days}日`;
    c.font = `600 ${f}px "Yu Gothic", sans-serif`;
    const bw = c.measureText(label).width + 44;
    box(c, w * .95 - bw, h * .045, bw, f * 1.85, 18, navy);
    c.fillStyle = '#ffffff'; c.fillText(label, w * .95 - bw + 22, h * .045 + f * 1.25);
  }
}
export function transition(c, w, h, effect, age) {
  if (effect === 'none') return;
  const progress = Math.min(1, Math.max(0, age / .7));
  c.save(); c.globalAlpha = effect === 'out' ? progress : 1 - progress;
  c.fillStyle = '#f2f5fa'; c.fillRect(0, 0, w, h); c.restore();
}
