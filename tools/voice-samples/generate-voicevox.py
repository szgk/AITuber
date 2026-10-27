"""公式規約で創作キャラへの声当てが許可されたVOICEVOX音声を比較する。"""
from pathlib import Path
import html
import io
import json
import urllib.parse
import urllib.request
import wave
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'voicevox-samples'
OUT.mkdir(parents=True, exist_ok=True)
BASE = 'http://127.0.0.1:50031'
TEXT = 'あ、あの……こんにちは。望月京子です。えっと、今日は……少しだけ、お話ししても、いいですか？'
RULES = {
    '冥鳴ひまり': 'https://www.meimeihimari.com/terms-of-use',
    '雨晴はう': 'https://amehau.com/rules/amehare-hau-rule',
    'ナースロボ＿タイプＴ': 'https://www.krnr.top/rules',
}

def request(route, params=None, body=None):
    url = BASE + route + ('?' + urllib.parse.urlencode(params) if params else '')
    data = json.dumps(body, ensure_ascii=False).encode('utf-8') if body is not None else None
    req = urllib.request.Request(url, data=data, method='POST' if route in ['/audio_query', '/synthesis'] else 'GET', headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()

speakers = json.loads(request('/speakers'))
version = json.loads(request('/version'))
plans = [
    ('01-himari', '冥鳴ひまり', 'ノーマル', '落ち着いた声・ゆっくり', .88, .80),
    ('02-hau', '雨晴はう', 'ノーマル', 'やわらかい声・ゆっくり', .88, .80),
    ('03-nurse-normal', 'ナースロボ＿タイプＴ', 'ノーマル', '控えめな声', .88, .80),
    ('04-nurse-fear', 'ナースロボ＿タイプＴ', '恐怖', '不安・震えの比較', .90, .78),
    ('05-nurse-whisper', 'ナースロボ＿タイプＴ', '内緒話', '小さな声の比較', .90, .82),
]
samples = []
for name, character, style, title, speed, intonation in plans:
    speaker = next(v for v in speakers if v['name'] == character)
    style_id = next(v['id'] for v in speaker['styles'] if v['name'] == style)
    query = json.loads(request('/audio_query', {'text': TEXT, 'speaker': style_id}))
    query.update(speedScale=speed, intonationScale=intonation, volumeScale=.95, prePhonemeLength=.22, postPhonemeLength=.32, outputSamplingRate=24000, outputStereo=False)
    raw = request('/synthesis', {'speaker': style_id}, query)
    (OUT / f'{name}.wav').write_bytes(raw)
    (OUT / f'{name}.query.json').write_text(json.dumps(query, ensure_ascii=False, indent=2), encoding='utf-8')
    with wave.open(io.BytesIO(raw), 'rb') as wav:
        seconds = wav.getnframes() / wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
    if seconds < 1 or int(np.max(np.abs(pcm.astype(np.int32)))) == 0:
        raise ValueError(f'音声生成の検証に失敗: {name}')
    samples.append(dict(file=f'{name}.wav', character=character, style=style, styleId=style_id, title=title, seconds=round(seconds, 2), credit=f'VOICEVOX:{character}', rules=RULES[character], processed=False))
    print(f'生成完了: {name} ({seconds:.1f}s)', flush=True)

# ナースロボの公式規約は出力音声の加工も明示的に許可している。
with wave.open(str(OUT / '03-nurse-normal.wav'), 'rb') as wav:
    params = wav.getparams()
    source = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float64) / 32768
t = np.arange(len(source)) / params.framerate
wobble = np.sin(2 * np.pi * 4.6 * t + .3 * np.sin(2 * np.pi * .7 * t))
positions = np.cumsum(1 + .002 * wobble) - 1
audio = np.interp(positions, np.arange(len(source)), source)
audio *= 1 - .055 * (.5 + .5 * np.sin(2 * np.pi * 4.1 * t))
with wave.open(str(OUT / '06-nurse-subtle-tremble.wav'), 'wb') as wav:
    wav.setparams(params); wav.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
samples.append({**samples[2], 'file': '06-nurse-subtle-tremble.wav', 'title': '小さな震えを追加', 'processed': True, 'processing': '音高の揺れ約0.2%、音量の揺れ最大5.5%。声質変換モデル・機械学習は使用しない。'})

cards = ''
for index, sample in enumerate(samples, 1):
    credit = html.escape(sample['credit'])
    cards += f'''<section><span class="number">{index:02d}</span><h2>{html.escape(sample['character'])} / {html.escape(sample['style'])}</h2><p>{html.escape(sample['title'])} · {sample['seconds']}秒{' · 後処理あり' if sample['processed'] else ' · 後処理なし'}</p><audio controls preload="metadata" src="{sample['file']}"></audio><div><a href="{sample['file']}" download>WAVを保存</a> · <a href="{sample['rules']}" target="_blank" rel="noopener">公式規約</a></div><small>{credit}</small></section>'''
page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>京子の声 / VOICEVOX比較</title><style>body{font-family:"Yu Gothic UI",sans-serif;background:#121621;color:#eeeaf4;max-width:860px;margin:45px auto;padding:22px;line-height:1.8}h1{font-size:28px}h2{font-size:18px;margin:0}section{background:#242a3a;border:1px solid #3c4357;border-radius:12px;padding:22px;margin:18px 0}audio{width:100%;margin:8px 0}a{color:#dfbdf5}.number{float:right;color:#b6a1ca}small{color:#b6bfd0}.notice{padding:18px;border-left:3px solid #cbb1e6;background:#202536}</style><h1>望月京子の声をえらぶ</h1><p>VOICEVOX · 創作キャラクターへの声当てを許可する音声から6種類を試作</p><p class="notice">台詞：あ、あの……こんにちは。望月京子です。えっと、今日は……少しだけ、お話ししても、いいですか？</p><p>01〜05はVOICEVOXの話速・抑揚の調整のみ。06は03に小さな震えを後処理で追加。同じ台詞で聞き比べてください。いずれも候補であり、本番の声は未採用です。</p>'''
page += cards + '''<p class="notice">公開時は使用した音声のクレジットと各規約を守ってください。ナースロボ＿タイプＴは企業の商用利用に事前連絡が必要です。音声とキャラクター画像の規約は別であり、本試作では公式画像は使用していません。</p><p><a href="https://voicevox.hiroshiba.jp/term/">VOICEVOX本体の規約</a> · 規約確認日：2026-10-23</p><script>document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(b!==a)b.pause()})))</script></html>'''
(OUT / 'index.html').write_text(page, encoding='utf-8')
(OUT / 'samples.json').write_text(json.dumps(dict(engine='VOICEVOX', version=version, checked='2026-10-23', text=TEXT, status='比較用・未採用', samples=samples), ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'CREDITS.txt').write_text('VOICEVOX\n' + '\n'.join(f'VOICEVOX:{name}\n{url}' for name, url in RULES.items()) + '\n\n公開時は使用した音声ごとの規約を守り、クレジットを掲載してください。\nナースロボ＿タイプＴの企業商用利用は連絡が必要です。\n', encoding='utf-8')
print('比較サンプル6件と試聴ページを生成しました。', flush=True)
