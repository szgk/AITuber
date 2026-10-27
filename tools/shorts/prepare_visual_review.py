"""実MP4から視覚レビュー用の証拠を抽出する。合格判定は付けない。"""
import argparse
import json
import math
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from render import ROOT, digest, find_binary

CHECKS = ['接地・床','遮蔽・前後関係','身体・姿勢','時間・カメラ','作画・光','指摘・画面']


def prepare(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    movie = ROOT/manifest['output']
    sha = digest(movie)
    destination = folder/'visual-review'/sha[:16]
    destination.mkdir(parents=True,exist_ok=True)
    duration,fps = manifest['duration'],manifest['fps']
    times = set()
    def add(t):
        times.add(round(max(0,min(duration-1/fps,t))*fps)/fps)
    for scene in manifest['timeline']:
        start,end = scene['start'],scene['end']
        for t in [start+.25,(start+end)/2,end-.25]:
            add(t)
        for key in scene.get('poseSequence',[]):
            t = start+key['at']*(end-start)
            for delta in [-1/fps,.07,.25]:
                add(t+delta)
        if start > .6:
            add(start-1/fps)
            add(start+.1)
    # 定期フレームは長い場面やカメラ移動の取りこぼしを補う。
    for t in range(1,math.ceil(duration),2):
        add(t)
    items,thumbnails,crops = [],[],[]
    font = ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',14)
    for i,t in enumerate(sorted(times)):
        file = destination/f'frame-{i:03}-{t:.3f}.png'
        subprocess.run([find_binary('ffmpeg'),'-hide_banner','-loglevel','error','-y','-ss',str(t),'-i',str(movie),'-frames:v','1',str(file)],check=True)
        with Image.open(file) as src:
            img = src.convert('RGB')
        thumb = img.resize((180,320))
        ImageDraw.Draw(thumb).text((4,4),f'{i:03} {t:.3f}s',font=font,fill='white',stroke_width=1,stroke_fill='black')
        thumbnails.append(thumb)
        # 机の下・靴底・床境界をまとめて拡大する。
        crop = img.crop((0,round(img.height*.42),round(img.width*.85),round(img.height*.84)))
        crop_file = destination/f'feet-{i:03}.png'
        crop.save(crop_file)
        crop.thumbnail((300,300))
        card = Image.new('RGB',(300,320),'#252837')
        card.paste(crop,(0,20))
        ImageDraw.Draw(card).text((4,2),f'{i:03} {t:.3f}s',font=font,fill='white')
        crops.append(card)
        items.append({'seconds':t,'frame':str(file.relative_to(folder)).replace('\\','/'),'feet':str(crop_file.relative_to(folder)).replace('\\','/')})
    sheets = []
    for label,cards,columns,per_sheet in [('overview',thumbnails,6,24),('feet',crops,4,12)]:
        for offset in range(0,len(cards),per_sheet):
            part = cards[offset:offset+per_sheet]
            cw,ch = part[0].size
            canvas = Image.new('RGB',(columns*cw,math.ceil(len(part)/columns)*ch),'#252837')
            for i,card in enumerate(part):
                canvas.paste(card,((i%columns)*cw,(i//columns)*ch))
            file = destination/f'{label}-{offset//per_sheet+1:02}.jpg'
            canvas.save(file,quality=92)
            sheets.append(str(file.relative_to(folder)).replace('\\','/'))
    evidence = {str(file.relative_to(folder)).replace('\\','/'):digest(file) for file in destination.iterdir() if file.suffix in ['.png','.jpg']}
    report = {'status':'未レビュー','output':str(movie.relative_to(ROOT)),'outputHash':sha,'method':'実MP4の時系列フレームと足元拡大を表示して確認する','sheets':sheets,'samples':items,'evidence':evidence,'inspectedEvidence':[],'checks':[{'name':name,'status':'未確認','evidence':[],'notes':''} for name in CHECKS],'findings':[]}
    target = folder/'visual-review.json'
    # 同じ対象に対する実施済み所見を再抽出だけで消さない。
    if target.exists():
        old = json.loads(target.read_text(encoding='utf-8'))
        if old.get('outputHash') == sha:
            report.update({k:old[k] for k in ['status','inspectedEvidence','checks','findings'] if k in old})
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'視覚レビュー証拠: {target}（判定: {report["status"]}）',flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/shorts/short-001-v2')
    args = parser.parse_args()
    prepare(ROOT/args.output)
