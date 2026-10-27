"""ローカル日本語音声から、控えめな声の揺れを比較する試作を生成する。"""
from pathlib import Path
import html
import json
import wave
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'voice-samples'
with wave.open(str(OUT / 'base-haruka.wav'), 'rb') as src:
    channels, width, rate, frames, _, _ = src.getparams()
    if width != 2:
        raise ValueError('16bit PCM音声が必要です')
    source = np.frombuffer(src.readframes(frames), dtype='<i2').reshape(-1, channels).astype(np.float64) / 32768

variants = [
    ('01-soft', 'A · 控えめ', .97, .0008, .025, .78, '震えを抑え、少しゆっくり。聞きやすさを優先した比較用。'),
    ('02-nervous', 'B · 少し緊張', .95, .0025, .065, .75, '小さな高さと音量の揺れ。気弱さと聞きやすさの中間。'),
    ('03-trembling', 'C · 震えを強めた比較用', .94, .0050, .13, .72, '震えを分かりやすくした比較用。長く聞くと揺れが目立つ可能性あり。'),
]
results = []
for name, title, speed, pitch_depth, amplitude_depth, gain, note in variants:
    count = int(len(source) / speed)
    time = np.arange(count) / rate
    # 不規則さを少し混ぜた4〜5Hzの揺れ。声優による感情演技とは区別する。
    wobble = np.sin(2 * np.pi * 4.6 * time + .3 * np.sin(2 * np.pi * .7 * time))
    positions = np.cumsum(speed * (1 + pitch_depth * wobble)) - speed
    envelope = gain * (1 - amplitude_depth * (.5 + .5 * np.sin(2 * np.pi * 4.1 * time)))
    audio = np.column_stack([np.interp(positions, np.arange(len(source)), source[:, ch]) for ch in range(channels)])
    audio *= envelope[:, None]
    audio = np.clip(audio, -1, 1)
    with wave.open(str(OUT / f'{name}.wav'), 'wb') as dst:
        dst.setnchannels(channels); dst.setsampwidth(2); dst.setframerate(rate)
        dst.writeframes((audio * 32767).astype('<i2').tobytes())
    results.append({'file': f'{name}.wav', 'title': title, 'seconds': round(count / rate, 2), 'peak': round(float(np.max(np.abs(audio))), 4), 'note': note, 'speed': speed, 'pitchDepth': pitch_depth, 'amplitudeDepth': amplitude_depth})

cards = ''.join(f'<section><h2>{html.escape(item["title"])}</h2><p>{html.escape(item["note"])}</p><audio controls preload="metadata" src="{item["file"]}"></audio><a href="{item["file"]}" download>WAVを保存</a></section>' for item in results)
(OUT / 'index.html').write_text('''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>京子の声・試作比較</title><style>body{font-family:"Yu Gothic UI",sans-serif;background:#141722;color:#ede8f4;max-width:760px;margin:50px auto;padding:20px;line-height:1.8}section{padding:20px;background:#242838;border-radius:12px;margin:20px 0}h2{font-size:19px}audio{width:100%}a{color:#d4b5ec}small{color:#b8bacb}</style><h1>京子の声 · 試作3種類</h1><p>「あ、あの……こんにちは。望月京子です。えっと、今日は……少しだけ、お話ししても、いいですか？」</p><p>同じ音声に小さな高さ・音量の揺れを加えた比較です。声・エンジンは未採用です。</p>''' + cards + '<small>ベース: この端末のMicrosoft Haruka。ローカル生成の方向性確認用です。自然な感情演技の完成サンプルではありません。公開動画で使う音声エンジンと利用条件は別途選定します。</small></html>', encoding='utf-8')
(OUT / 'samples.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(results, ensure_ascii=True, indent=2))
