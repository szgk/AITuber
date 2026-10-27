import { background as drawBackground, caption, overlay, transition } from './visuals.js';
const $ = id => document.getElementById(id);
const canvas = $('stage'), ctx = canvas.getContext('2d');
const output = new URLSearchParams(location.search).has('output');
document.body.classList.toggle('output', output);
const controls = ['format', 'shot', 'background', 'scale', 'motion', 'demo', 'sensitivity', 'script', 'captions', 'captionStyle', 'title', 'days', 'branding', 'effect'];
const logo = new Image(); logo.src = '../../docs/character-design/logo/kyoko-logo.png';
let effectStart = performance.now();
$('replay').onclick = () => { effectStart = performance.now(); channel?.postMessage({ replay: true }); };
$('still').onclick = () => { const a = document.createElement('a'); a.download = 'kyoko-frame.png'; a.href = canvas.toDataURL('image/png'); a.click(); };
$('backgroundPng').onclick = () => {
  if (!['soft', 'night', 'paper'].includes($('background').value)) { status('背景PNGは今回追加した3種類の背景から選んでください。'); return; }
  const sheet = document.createElement('canvas'); sheet.width = canvas.width; sheet.height = canvas.height;
  drawBackground(sheet.getContext('2d'), sheet.width, sheet.height, $('background').value);
  const a = document.createElement('a'); a.download = `kyoko-background-${$('background').value}-${sheet.width}x${sheet.height}.png`; a.href = sheet.toDataURL(); a.click();
};
let expression = 'normal', line = 0, mouth = 0, avatar, analyser, micStream, audioContext, mix, micNode, fileNode;
let recorder, capture, chunks = [], fileHandle, writer, writeQueue, writeError, startTime = 0, pausedAt = 0, pauseDuration = 0;
let audioUrl, micPending = false;
const channel = typeof BroadcastChannel === 'function' ? new BroadcastChannel('kyoko-studio') : null;
function status(message) { $('status').textContent = message; }
function settings() { return Object.fromEntries(controls.map(id => [id, $(id).type === 'checkbox' ? $(id).checked : $(id).value])); }
function publish() { if (!output) channel?.postMessage({ settings: settings(), expression, line, mouth }); }
channel && (channel.onmessage = ({ data }) => {
  if (data.replay && output) { effectStart = performance.now(); return; }
  if (data.request) { publish(); return; }
  if (!output) return;
  if (data.settings?.effect !== $('effect').value) effectStart = performance.now();
  for (const [id, value] of Object.entries(data.settings || {})) $(id)[$(id).type === 'checkbox' ? 'checked' : 'value'] = value;
  expression = data.expression; line = data.line; mouth = data.mouth; resize();
});
if (output) channel?.postMessage({ request: true });
function resize() {
  const portrait = $('format').value === 'portrait';
  const square = $('format').value === 'square';
  const w = portrait || square ? 1080 : 1920, h = portrait ? 1920 : 1080;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  $('dimensions').textContent = `${w} × ${h} / 30fps`;
}
for (const id of controls) $(id).addEventListener('input', () => { if (id === 'script') line = 0; if (id === 'effect') effectStart = performance.now(); resize(); publish(); });
function selectExpression(value) {
  expression = value;
  document.querySelectorAll('[data-expression]').forEach(b => b.classList.toggle('selected', b.dataset.expression === value));
  publish();
}
document.querySelectorAll('[data-expression]').forEach(b => b.onclick = () => selectExpression(b.dataset.expression));
function moveLine(delta) { line = Math.max(0, Math.min($('script').value.split('\n').length - 1, line + delta)); publish(); }
$('previous').onclick = () => moveLine(-1); $('next').onclick = () => moveLine(1);
document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(document.activeElement.tagName)) return;
  if (['1', '2', '3'].includes(e.key)) selectExpression(['normal', 'smile', 'surprise'][Number(e.key) - 1]);
  if (e.key === 'ArrowRight') moveLine(1); if (e.key === 'ArrowLeft') moveLine(-1);
});
// 外周から連続する明るい背景だけを除去し、衣装の白を保持する。
async function loadAvatar() {
  const image = new Image(); image.src = '../../docs/character-design/kyoko-design-ver1.png'; await image.decode();
  const cut = document.createElement('canvas'); cut.width = 320; cut.height = 875;
  const c = cut.getContext('2d', { willReadFrequently: true }); c.drawImage(image, 126, 119, 320, 875, 0, 0, 320, 875);
  const pixels = c.getImageData(0, 0, 320, 875), { data } = pixels;
  const seen = new Uint8Array(320 * 875), queue = new Int32Array(320 * 875); let head = 0, tail = 0;
  const add = p => { if (p < 0 || p >= seen.length || seen[p]) return; seen[p] = 1; const i = p * 4;
    if (Math.min(data[i], data[i + 1], data[i + 2]) > 190) queue[tail++] = p; };
  for (let x = 0; x < 320; x++) { add(x); add(874 * 320 + x); }
  for (let y = 0; y < 875; y++) { add(y * 320); add(y * 320 + 319); }
  while (head < tail) { const p = queue[head++]; data[p * 4 + 3] = 0;
    if (p % 320) add(p - 1); if (p % 320 < 319) add(p + 1); add(p - 320); add(p + 320); }
  // デザインシートの見出し・配色見本など、人物と離れた断片を除く。
  const labels = new Int32Array(seen.length); let component = 0, largest = 0, largestSize = 0;
  for (let seed = 0; seed < labels.length; seed++) {
    if (labels[seed] || !data[seed * 4 + 3]) continue;
    component++; head = 0; tail = 0; queue[tail++] = seed; labels[seed] = component;
    const visit = p => { if (p < 0 || p >= labels.length || labels[p] || !data[p * 4 + 3]) return; labels[p] = component; queue[tail++] = p; };
    while (head < tail) { const p = queue[head++]; if (p % 320) visit(p - 1); if (p % 320 < 319) visit(p + 1); visit(p - 320); visit(p + 320); }
    if (tail > largestSize) { largestSize = tail; largest = component; }
  }
  for (let p = 0; p < labels.length; p++) if (labels[p] !== largest) data[p * 4 + 3] = 0;
  c.putImageData(pixels, 0, 0); avatar = cut; $('record').disabled = output || !window.MediaRecorder;
  $('badge').textContent = '● スタジオ準備完了'; status('音声ファイルを読み込んで収録開始。マイクは不要です。無音でも収録できます。');
}
function ellipse(x, y, rx, ry, fill) { ctx.beginPath(); ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2); ctx.fillStyle = fill; ctx.fill(); }
function draw(t) {
  const w = canvas.width, h = canvas.height, bg = $('background').value;
  ctx.fillStyle = ({green:'#00b140', white:'#ffffff', dark:'#141a2a'})[bg] || '#e9e6ef'; ctx.fillRect(0, 0, w, h);
  if (bg === 'room') {
    const g = ctx.createLinearGradient(0, 0, w, h); g.addColorStop(0, '#f2eaf0'); g.addColorStop(1, '#b8c5d8'); ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#ffffff70'; ctx.lineWidth = 2;
    for (let x = 80; x < w; x += 180) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    ellipse(w * .77, h * .22, w * .13, w * .13, '#fff8');
    ctx.fillStyle = '#43516a'; ctx.font = `${Math.round(w * .018)}px sans-serif`; ctx.fillText('MOCHIZUKI KYOKO', w * .06, h * .07);
  }
  if (['soft', 'night', 'paper'].includes(bg)) drawBackground(ctx, w, h, bg);
  if (avatar) {
    const animate = $('motion').checked, bust = $('shot').value === 'bust';
    const s = Math.min(w / 420, h / (bust ? 410 : 1030)) * Number($('scale').value);
    ctx.save(); ctx.translate(w / 2, bust ? h * .11 : h * .12);
    if (animate) { ctx.translate(Math.sin(t * .7) * 4, Math.sin(t * 1.8) * 3); ctx.rotate(Math.sin(t * .65) * .012); }
    ctx.scale(s, s); ctx.translate(-160, 0); ctx.drawImage(avatar, 0, 0);
    // 原画の口を肌色で覆い、音量と表情から口形状を合成する。
    ellipse(166, 155, 14, 9, '#faf9f7');
    ctx.strokeStyle = '#292a2b'; ctx.lineWidth = 2; ctx.lineCap = 'round'; ctx.beginPath();
    if (mouth > .08 || expression === 'surprise') {
      const open = Math.max(mouth, expression === 'surprise' ? .45 : 0);
      ellipse(166, 155, 5 + open * 3, 2 + open * 11, '#3f222b'); ellipse(166, 158 + open * 4, 3 + open * 2, 1 + open * 3, '#eaa2aa');
    } else if (expression === 'smile') { ctx.moveTo(155, 153); ctx.quadraticCurveTo(166, 165, 176, 151); ctx.stroke(); }
    else { ctx.moveTo(155, 155); ctx.lineTo(160, 152); ctx.lineTo(165, 156); ctx.lineTo(171, 153); ctx.lineTo(177, 155); ctx.stroke(); }
    const blink = animate && (t % 4.7 > 4.52);
    if (blink) { for (const [x,y] of [[130,130],[189,119]]) { ctx.save(); ctx.translate(x,y); ctx.rotate(-.17); ellipse(0,0,13,9,'#faf9f7'); ctx.beginPath(); ctx.moveTo(-12,0); ctx.quadraticCurveTo(0,5,12,-1); ctx.stroke(); ctx.restore(); } }
    ctx.restore();
  }
  if ($('captions').checked && $('captionStyle').value === 'classic') {
    const text = $('script').value.split('\n')[line] || ''; const font = w > h ? 46 : 51;
    ctx.font = `600 ${font}px "Yu Gothic",sans-serif`; const lines = []; let current = '';
    for (const char of text) { if (ctx.measureText(current + char).width > w * .84) { lines.push(current); current = ''; } current += char; } if (current) lines.push(current);
    const visible = lines.slice(0, 4);
    if (visible.length) { const height = visible.length * font * 1.5 + 30, y = h - height - h * .065;
      ctx.fillStyle = '#121827dc'; ctx.fillRect(w * .05, y, w * .9, height); ctx.fillStyle = '#fff'; ctx.textAlign = 'center';
      visible.forEach((value, i) => ctx.fillText(value, w / 2, y + font * (1.3 + i * 1.5))); ctx.textAlign = 'left'; }
  }
  if ($('captions').checked && $('captionStyle').value !== 'classic') caption(ctx, w, h, $('script').value.split('\n')[line] || '', $('captionStyle').value);
  overlay(ctx, w, h, $('title').value, $('days').value, $('branding').checked ? logo : null);
  transition(ctx, w, h, $('effect').value, (performance.now() - effectStart) / 1000);
}
let lastPublish = 0;
function frame(ms) {
  const t = ms / 1000;
  if (!output) {
    let level = 0;
    if (analyser) { const values = new Float32Array(analyser.fftSize); analyser.getFloatTimeDomainData(values); level = Math.sqrt(values.reduce((sum, v) => sum + v * v, 0) / values.length); }
    $('level').value = Math.min(1, level * 5);
    const target = $('demo').checked ? Math.max(0, Math.sin(t * 14)) * .7 : Math.min(1, Math.max(0, level - .008) * Number($('sensitivity').value) * 3);
    mouth += (target - mouth) * .4;
    if (ms - lastPublish > 80) { publish(); lastPublish = ms; }
  }
  draw(t);
  if (recorder && recorder.state !== 'inactive') {
    const elapsed = Math.max(0, Math.floor(((pausedAt || performance.now()) - startTime - pauseDuration) / 1000));
    $('timer').textContent = [Math.floor(elapsed / 3600), Math.floor(elapsed / 60) % 60, elapsed % 60].map(n => String(n).padStart(2, '0')).join(':');
  }
  requestAnimationFrame(frame);
}
async function setupAudio() {
  if (!audioContext) { audioContext = new AudioContext(); analyser = audioContext.createAnalyser(); analyser.fftSize = 1024; mix = audioContext.createMediaStreamDestination(); }
  await audioContext.resume();
}
$('mic').onclick = async () => {
  if (micPending) return;
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micNode.disconnect(); micStream = null; $('mic').textContent = 'マイクを接続'; return; }
  micPending = true; $('mic').disabled = true;
  try { await setupAudio(); micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true }, video: false });
    micNode = audioContext.createMediaStreamSource(micStream); micNode.connect(analyser); micNode.connect(mix); $('mic').textContent = 'マイクを切断'; status('マイク接続済み。声に合わせて口が動きます。');
  } catch (e) { status(`マイクを接続できません: ${e.message}`); } finally { micPending = false; $('mic').disabled = false; }
};
$('audioFile').onchange = async () => {
  const file = $('audioFile').files[0]; if (!file) return;
  try { await setupAudio(); $('audio').pause(); if (audioUrl) URL.revokeObjectURL(audioUrl); audioUrl = URL.createObjectURL(file); $('audio').src = audioUrl;
    if (!fileNode) { fileNode = audioContext.createMediaElementSource($('audio')); fileNode.connect(analyser); fileNode.connect(mix); fileNode.connect(audioContext.destination); }
    status('音声を読み込みました。収録開始時に先頭から再生します。');
  } catch (e) { status(`音声を読み込めません: ${e.message}`); }
};
function lock(active) { for (const id of ['format','record','audioFile']) $(id).disabled = active; $('pause').disabled = !active; $('stop').disabled = !active; }
$('record').onclick = async () => {
  lock(true); chunks = []; fileHandle = null; writer = null; writeError = null; writeQueue = Promise.resolve();
  try {
    const mime = ['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm','video/mp4'].find(type => MediaRecorder.isTypeSupported(type));
    if (!mime) throw new Error('このブラウザは対応する録画形式を持ちません。ChromeまたはEdgeをご利用ください。');
    const extension = mime.includes('mp4') ? 'mp4' : 'webm', name = `kyoko-${new Date().toISOString().replace(/[:.]/g,'-')}.${extension}`;
    if ('showSaveFilePicker' in window) {
      fileHandle = await window.showSaveFilePicker({ suggestedName: name, types: [{ description: '収録動画', accept: { [mime.split(';')[0]]: [`.${extension}`] } }] });
      writer = await fileHandle.createWritable();
    }
    await setupAudio(); capture = canvas.captureStream(30); mix.stream.getAudioTracks().forEach(track => capture.addTrack(track.clone()));
    recorder = new MediaRecorder(capture, { mimeType: mime, videoBitsPerSecond: 8000000 });
    recorder.ondataavailable = ({ data }) => { if (!data.size) return; if (writer) writeQueue = writeQueue.then(() => writer.write(data)).catch(e => { writeError = e; if (recorder.state !== 'inactive') recorder.stop(); }); else chunks.push(data); };
    recorder.onstop = async () => {
      $('audio').pause(); capture.getTracks().forEach(track => track.stop());
      try { await writeQueue; if (writeError) throw writeError;
        if (writer) { await writer.close(); status(`保存しました: ${name}`); }
        else { const blob = new Blob(chunks, { type: mime }), url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = name; link.textContent = `${name} を保存`; $('downloads').append(link); link.click(); status('収録完了。保存リンクから再ダウンロードできます。'); chunks = []; }
      } catch (e) { await writer?.abort().catch(() => {}); status(`保存に失敗しました: ${e.message}`); }
      finally { lock(false); $('badge').textContent = '● スタジオ準備完了'; $('pause').textContent = '一時停止'; }
    };
    recorder.onerror = e => { status(`録画エラー: ${e.error?.message || '録画を停止します'}`); if (recorder.state !== 'inactive') recorder.stop(); };
    startTime = performance.now(); effectStart = startTime; line = 0; channel?.postMessage({ replay: true }); pausedAt = 0; pauseDuration = 0; recorder.start(1000); $('badge').textContent = '● REC';
    if ($('audio').src) { $('audio').currentTime = 0; await $('audio').play().catch(e => status(`録画中。音声は手動再生してください: ${e.message}`)); }
    status(writer ? '収録中。動画をディスクへ順次書き込みます。' : '収録中。メモリ保存方式です。長時間収録はChrome/Edgeのファイル保存またはOBSをご利用ください。');
  } catch (e) { capture?.getTracks().forEach(t => t.stop()); await writer?.abort().catch(() => {}); lock(false); status(e.name === 'AbortError' ? '収録をキャンセルしました。' : `収録を開始できません: ${e.message}`); }
};
let resumeAudio = false;
$('pause').onclick = async () => {
  if (recorder?.state === 'recording') { recorder.pause(); pausedAt = performance.now(); resumeAudio = !$('audio').paused; $('audio').pause(); $('pause').textContent = '再開'; $('badge').textContent = 'Ⅱ PAUSED'; }
  else if (recorder?.state === 'paused') { pauseDuration += performance.now() - pausedAt; pausedAt = 0; recorder.resume(); if (resumeAudio) await $('audio').play().catch(e => status(e.message)); $('pause').textContent = '一時停止'; $('badge').textContent = '● REC'; }
};
$('stop').onclick = () => { if (recorder && recorder.state !== 'inactive') { $('stop').disabled = true; $('pause').disabled = true; recorder.stop(); } };
window.addEventListener('beforeunload', e => { if (recorder && recorder.state !== 'inactive') { e.preventDefault(); e.returnValue = ''; } });
document.addEventListener('visibilitychange', () => { if (document.hidden && recorder?.state === 'recording') status('収録中は画面を表示してください。非表示のタブでは映像更新が制限されます。'); });
resize(); requestAnimationFrame(frame); loadAvatar().catch(e => { $('badge').textContent = '素材エラー'; status(`原画の読み込みに失敗しました。npm run studio で起動してください: ${e.message}`); });
