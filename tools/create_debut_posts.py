"""既存デザインを使った告知カードと対応原稿を保存する。"""
from pathlib import Path
import re, json, hashlib
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/debut-posts'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'images').mkdir(exist_ok=True)
(OUT / 'texts').mkdir(exist_ok=True)
(OUT / 'review').mkdir(exist_ok=True)
navy = '#233449'
font = lambda size: ImageFont.truetype('C:/Windows/Fonts/YuGothB.ttc', size)
source = Image.open(ROOT / 'docs/character-design/kyoko-design-ver1.png').convert('RGB')
# 原画の単独人物パネル。設定表の枠・見出し・色見本を含めない。
body = source.crop((133, 115, 447, 990))
ImageDraw.Draw(body).rectangle((0,0,44,55),fill='white')
smile = source.crop((835, 442, 1154, 684))
titles = ['はじめまして', '昨日より、少し', 'ご挨拶の練習中', 'かわいいものが好き', 'サーモンが好き', '笑ったところだけ', 'ゲームの話なら', 'あと7日', '顔を見て、こんばんは', '大きな犬が好き', 'お洋服も、少しだけ', 'のんびり、お話し', 'どちらから言おう', '明日、お会いできたら', 'ちゃんと、はじめまして']
plan = (ROOT / 'docs/debut-tweet-plan.md').read_text(encoding='utf-8')
parts = re.split(r'### ', plan)[1:16]
posts = []
for i, part in enumerate(parts):
    day = 14-i
    key = f'd-{day:02d}' if day else 'd-00'
    text = '\n'.join(line[2:] for line in part.splitlines() if line.startswith('> '))
    if day == 12:
        text = '顔を出す練習を、しています。\n「こんにちは」だけでも、何度も言い直してしまって……。\n今日は、ここまで。もう少しずつ、がんばります。'
    if day == 7:
        text = 'あと7日で、デビューです。\n今日は、片目だけ……。\n目が合うと緊張しますね。画面なのに。不思議です。'
    if day == 9:
        text = text.replace('今のところも、映ってました？', '今のところも、見えてました？')
    if day == 4:
        text = text.replace('映るのは一瞬', '見えるのは少し')
    im = Image.new('RGB', (1200,1200), '#f7f7f3')
    d = ImageDraw.Draw(im)
    d.rectangle((0,0,1200,18), fill=navy)
    d.text((68,55), '望月京子  /  デビューまでのご挨拶', font=font(26), fill=navy)
    d.text((68,108), titles[i], font=font(48), fill=navy)
    label = f'デビューまで {day} 日' if day else '本日デビュー'
    d.rounded_rectangle((68,209,430,269), radius=30, fill=navy)
    d.text((94,220), label, font=font(28), fill='white')
    # 白い窓に部分図を置く。家具・床として描かないグラフィック構成。
    box=(68,310,1132,1060)
    d.rounded_rectangle(box, radius=28, fill='white')
    if day >= 10:
        crop=body.crop((0,0,314, 62 if day==14 else 76 if day==13 else 84 if day==12 else 92 if day==11 else 100))
        crop=ImageOps.contain(crop,(760,280))
        im.paste(crop,((1200-crop.width)//2,790-crop.height))
        d=ImageDraw.Draw(im)
        d.rectangle((100,790,1100,1020),fill=navy)
        note=['あ、あの……', '少しずつ、前へ。', 'こんにちは、の練習。', '色も、形も、裏側も。', 'ここは、迷いません。'][i]
        d.text((150,855),note,font=font(36),fill='white')
    elif day in (9,8):
        crop=smile.crop((0,164,319,242)).resize((957,234))
        im.paste(crop,(121,490))
        d=ImageDraw.Draw(im)
        d.text((145,850),'少しずつ、わたしのこと。',font=font(34),fill=navy)
    elif day==7:
        crop=smile.crop((0,0,180,242)).resize((540,726))
        im.paste(crop,(320,322))
        d=ImageDraw.Draw(im)
        d.rectangle((320,322,550,1048),fill=navy)
    elif day in (6,5):
        crop=ImageOps.contain(smile,(920,690))
        im.paste(crop,((1200-crop.width)//2,340))
    elif day in (4,3,2):
        crop=ImageOps.contain(body.crop((0,0,314,500)),(700,710))
        im.paste(crop,((1200-crop.width)//2,330))
    else:
        crop=ImageOps.contain(body,(500,710))
        im.paste(crop,(720,330))
        logo=Image.open(ROOT/'docs/character-design/logo/kyoko-logo.png').convert('RGBA')
        logo.thumbnail((540,270))
        im.paste(logo,(130,490),logo)
        d=ImageDraw.Draw(im)
        d.text((135,800),'あ、あの……\nよろしくお願いします。',font=font(28),fill=navy,spacing=14)
    d=ImageDraw.Draw(im)
    d.text((68,1100),'MOCHIZUKI KYOKO',font=font(25),fill=navy)
    d.text((960,1100),f'{i+1:02d} / 15',font=font(25),fill=navy)
    path=OUT/'images'/f'{key}.png'
    im.save(path)
    alt=f'望月京子の告知カード。「{titles[i]}」「{label}」。'
    alt+= ('頭頂部のみが紺の図形の上に見える。' if day>=10 else '頬と口元の部分図。目は画面外。' if day in (9,8) else '縦の紺の図形の横から片目を見せる。' if day==7 else '顔全体のイラスト。' if day in (6,5) else '上半身のイラスト。' if day in (4,3,2) else '全身イラストと望月京子のロゴ。')
    (OUT/'texts'/f'{key}.txt').write_text(text+'\n',encoding='utf-8')
    posts.append(dict(day=f'D−{day}' if day else 'D当日',text=text,image=f'images/{key}.png',alt=alt))
    if day<=1:
        im.crop((740,850,1090,1060)).resize((1000,600)).save(OUT/'review'/f'{key}-feet.png')
sheet=Image.new('RGB',(1500,900),'#ddd')
for i,p in enumerate(posts):
    thumb=Image.open(OUT/p['image']).resize((300,300))
    sheet.paste(thumb,((i%5)*300,(i//5)*300))
sheet.save(OUT/'review/contact-sheet.png')
(OUT/'posts.json').write_text(json.dumps(posts,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 望月京子：デビュー14日前から当日までの投稿原稿', '', 'D＝デビュー日（未定）。1日1投稿・公開日割りは提案。外部投稿は未実施。', '', '添付PNGは1200×1200の静止画。既存計画の動作・手のポーズ・動画は再現していないため、文面は実画像に合わせて調整した。D−14の動く頭、D−7以降の音声付き動画は別途制作対象。デビュー日・時刻・公開先URLは未確定で、この原稿には架空の値を入れていない。公開前に確定情報とURLを追記する。', '']
for p in posts:
    lines += [f"## {p['day']}", '',p['text'],'',f"![{p['alt']}]({p['image']})",'',f"代替テキスト：{p['alt']}",'']
(OUT/'投稿原稿.md').write_text('\n'.join(lines),encoding='utf-8')
print(str(OUT))
