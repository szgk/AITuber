// 同梱PlaywrightまたはローカルインストールのPlaywrightで実行する。
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
(async () => {
  const server = require('node:child_process').spawn(process.execPath, [require('node:path').join(__dirname, 'server.mjs')], { env: { ...process.env, PORT: '4193' }, stdio: ['ignore', 'pipe', 'pipe'] });
  await new Promise((resolve, reject) => { server.stdout.once('data', resolve); server.once('error', reject); server.once('exit', code => reject(new Error(`サーバー終了: ${code}`))); });
  const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] });
  try {
    const context = await browser.newContext({ permissions: ['microphone'], acceptDownloads: true });
    const page = await context.newPage(); const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.addInitScript(() => { delete window.showSaveFilePicker; });
    await page.goto('http://127.0.0.1:4193/frontend/studio/');
    await page.waitForFunction(() => !document.getElementById('record').disabled);
    const wave = Buffer.alloc(44 + 48000 * 2); wave.write('RIFF'); wave.writeUInt32LE(wave.length - 8, 4); wave.write('WAVEfmt ', 8); wave.writeUInt32LE(16, 16); wave.writeUInt16LE(1, 20); wave.writeUInt16LE(1, 22); wave.writeUInt32LE(48000, 24); wave.writeUInt32LE(96000, 28); wave.writeUInt16LE(2, 32); wave.writeUInt16LE(16, 34); wave.write('data', 36); wave.writeUInt32LE(96000, 40);
    for (let i = 0; i < 48000; i++) wave.writeInt16LE(Math.round(Math.sin(i * 440 * Math.PI * 2 / 48000) * 6000), 44 + i * 2);
    await page.locator('#audioFile').setInputFiles({ name: '音声テスト.wav', mimeType: 'audio/wav', buffer: wave });
    await page.waitForFunction(() => document.getElementById('audio').readyState >= 2);
    await page.locator('#script').fill('あ、あの……こんにちは。\n望月京子です。');
    await page.locator('#demo').check();
    await page.locator('#record').click();
    await page.waitForTimeout(1400); await page.locator('#pause').click();
    if (await page.locator('#pause').textContent() !== '再開') throw new Error('一時停止に失敗');
    await page.locator('#pause').click(); await page.locator('#next').click();
    const download = page.waitForEvent('download'); await page.locator('#stop').click(); const saved = await download;
    if ((await saved.failure()) !== null) throw new Error('動画保存に失敗');
    const fs = require('node:fs'); const size = fs.statSync(await saved.path()).size; if (size < 1000) throw new Error('動画が空です');
    await page.waitForFunction(() => !document.getElementById('record').disabled);
    await page.locator('#format').selectOption('landscape');
    const dimensions = await page.locator('#stage').evaluate(c => [c.width, c.height]);
    if (dimensions.join() !== '1920,1080') throw new Error('横画面の寸法不正');
    const out = await context.newPage(); await out.goto('http://127.0.0.1:4193/frontend/studio/?output=1');
    await out.waitForFunction(() => document.getElementById('stage').width === 1920);
    await page.bringToFront();
    await page.locator('#shot').selectOption('bust');
    const assetDir = require('node:path').join(__dirname, '../../docs/character-design/video-kit');
    fs.mkdirSync(assetDir, { recursive: true });
    await page.locator('#branding').check();
    await page.locator('#title').fill('はじめまして');
    await page.locator('#days').selectOption('7');
    for (const format of ['portrait', 'landscape', 'square']) {
      await page.locator('#format').selectOption(format);
      for (const bg of ['soft', 'night', 'paper']) {
        await page.locator('#background').selectOption(bg);
        await page.waitForTimeout(120);
        const png = page.waitForEvent('download'); await page.locator('#backgroundPng').click();
        const file = await png; await file.saveAs(require('node:path').join(assetDir, file.suggestedFilename()));
      }
      await page.locator('#background').selectOption('soft');
      await page.waitForTimeout(120);
      await page.locator('#stage').screenshot({ path: require('node:path').join(assetDir, `preview-${format}.png`) });
    }
    await page.locator('#effect').selectOption('out');
    await page.locator('#replay').click(); await page.waitForTimeout(850);
    const faded = await page.locator('#stage').evaluate(c => Array.from(c.getContext('2d').getImageData(c.width / 2, c.height / 2, 1, 1).data));
    if (faded.slice(0, 3).join() !== '242,245,250') throw new Error('退場フェードが完了しません');
    await page.locator('#effect').selectOption('in'); await page.locator('#replay').click();
    await page.waitForTimeout(850);
    const visible = await page.locator('#stage').evaluate(c => Array.from(c.getContext('2d').getImageData(c.width / 2, c.height / 2, 1, 1).data));
    if (visible.join() === faded.join()) throw new Error('登場フェードが完了しません');
    await out.waitForFunction(() => document.getElementById('days').value === '7' && document.getElementById('stage').height === 1080 && document.getElementById('stage').width === 1080);
    await page.screenshot({ path: '.cache/kyoko-studio.png', fullPage: true });
    if (errors.length) throw new Error(errors.join('\n'));
    console.log(`成功: 原画読込、音声、字幕、録画・保存(${size} bytes)、3画角、背景9枚PNG、ロゴ、告知表示、登場・退場、OBS同期`);
  } finally { await browser.close(); server.kill(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
