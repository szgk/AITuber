"""台本・専用ポーズ・空間レイヤーから音声付きショートを再現可能に制作する。"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from lipsync import phoneme_cues,mouth_at,mouth_only

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_binary(name):
    explicit = os.environ.get(name.upper())
    if explicit:
        return explicit
    installed = shutil.which(name)
    if installed:
        return installed
    if name == 'ffmpeg':
        local = list((ROOT / '.cache/shorts-python/imageio_ffmpeg/binaries').glob('ffmpeg*.exe'))
        if local:
            return str(local[0])
    # 参考プロジェクトの既存バイナリを読み取り専用で利用する。
    for base in [ROOT / 'node_modules', ROOT.parent / 'auto_movie/video-generator/node_modules']:
        matches = list(base.glob(f'.pnpm/@remotion+compositor-win32*/node_modules/@remotion/compositor-win32*/{name}.exe'))
        if matches:
            return str(matches[0])
    raise RuntimeError(f'{name}がありません。環境変数{name.upper()}で実行ファイルを指定してください。')


def api(base, route, params=None, body=None):
    url = base + route + ('?' + urllib.parse.urlencode(params) if params else '')
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method='POST' if route in ['/audio_query', '/synthesis'] else 'GET', headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def save_wav(path, pcm, rate=24000):
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes())


def make_audio(story, voice, out, base):
    version = json.loads(api(base, '/version'))
    speaker = next(s for s in json.loads(api(base, '/speakers')) if s['speaker_uuid'] == voice['speakerUuid'])
    style = next(s for s in speaker['styles'] if s['name'] == voice['style'])
    rate = voice['query']['outputSamplingRate']
    chunks, cursor = [np.zeros(round(.55 * rate))], .55
    timeline = []
    for index, scene in enumerate(story['scenes']):
        speech_text = scene.get('speechText',scene['text'])
        key = hashlib.sha256(json.dumps([speech_text, voice, version, style], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache = out / 'audio' / f'{key}.wav'
        cache.parent.mkdir(exist_ok=True)
        if not cache.exists():
            query = json.loads(api(base, '/audio_query', {'text': speech_text, 'speaker': style['id']}))
            query.update(voice['query'])
            raw = api(base, '/synthesis', {'speaker': style['id']}, query)
            with wave.open(io.BytesIO(raw)) as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != rate:
                    raise ValueError('音声形式が採用設定と異なります')
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(float) / 32768
            t = np.arange(len(pcm)) / rate
            p = voice['processing']
            wobble = np.sin(2*np.pi*p['pitchFrequencyHz']*t + p['pitchPhaseModulationDepth']*np.sin(2*np.pi*p['pitchPhaseModulationFrequencyHz']*t))
            pcm = np.interp(np.cumsum(1+p['pitchDepth']*wobble)-1, np.arange(len(pcm)), pcm)
            pcm *= 1-p['amplitudeDepth']*(.5+.5*np.sin(2*np.pi*p['amplitudeFrequencyHz']*t))
            save_wav(cache, pcm, rate)
            cache.with_suffix('.query.json').write_text(json.dumps(query, ensure_ascii=False, indent=2), encoding='utf-8')
        with wave.open(str(cache)) as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(float) / 32768
        if not len(pcm) or np.max(np.abs(pcm)) < .001:
            raise ValueError('音声が空または無音です')
        duration = len(pcm)/rate
        query = json.loads(cache.with_suffix('.query.json').read_text(encoding='utf-8'))
        cues = phoneme_cues(query,duration,cursor)
        timeline.append({**scene, 'start': cursor, 'speechEnd': cursor+duration, 'end':cursor+duration+scene['pauseAfter'], 'audio':str(cache.relative_to(ROOT)),'mouthCues':cues,'query':str(cache.with_suffix('.query.json').relative_to(ROOT))})
        chunks.extend([pcm, np.zeros(round(scene['pauseAfter']*rate))])
        cursor = sum(len(c) for c in chunks) / rate
        print(f'音声 {index+1}/{len(story["scenes"])}: {duration:.2f}秒', flush=True)
    pcm = np.concatenate(chunks)
    # 原創の控えめな短いベル。声のない冒頭と終盤の余韻にのみ配置。
    for start, freq in [(.08,659.25),(.27,783.99),(cursor-1.0,659.25),(cursor-.7,987.77)]:
        n = min(round(.65*rate), len(pcm)-round(start*rate))
        t = np.arange(n)/rate
        bell = .025*np.sin(2*np.pi*freq*t)*np.exp(-7*t)*np.minimum(t/.01,1)
        offset = round(start*rate)
        pcm[offset:offset+n] += bell
    save_wav(out/'soundtrack.wav', pcm, rate)
    return timeline, cursor, pcm, {'version':version,'speaker':speaker['name'],'style':style}


def ease(v):
    v = min(1, max(0, v))
    return v*v*(3-2*v)


class Stage:
    def __init__(self, story, width, height):
        self.story, self.w, self.h = story, width, height
        self.bg = ImageOps.fit(Image.open(ROOT/story['background']).convert('RGB'), (width,height), method=Image.Resampling.LANCZOS).convert('RGBA')
        self.poses = {}
        for name, file in story['poses'].items():
            image = Image.open(ROOT/file).convert('RGBA')
            alpha = image.getchannel('A')
            if alpha.getextrema()[0] == 255:
                raise ValueError(f'{name}に実際の透過がありません')
            # 低アルファの生成時の柔らかい縁を境界に含めすぎない。
            bbox = alpha.point(lambda p: 255 if p > 24 else 0).getbbox()
            self.poses[name] = image.crop(bbox)
        self.mask = Image.new('L',(width,height))
        ImageDraw.Draw(self.mask).polygon([(round(x*width),round(y*height)) for x,y in story['foregroundPolygon']],fill=255)
        for polygon in story.get('foregroundObjects',[]):
            ImageDraw.Draw(self.mask).polygon([(round(x*width),round(y*height)) for x,y in polygon],fill=255)
        # ライトの細い支柱・曲線も背景原画の輪郭から復元し、髪より前に置く。
        for item in story.get('foregroundSilhouettes',[]):
            original = Image.open(ROOT/story['background']).convert('RGB')
            ow,oh = original.size
            box = tuple(round(v*(ow if i%2==0 else oh)) for i,v in enumerate(item['roi']))
            region = np.array(original.crop(box))
            dark = (region.max(axis=2)<item.get('threshold',175)).astype(np.uint8)*255
            shape = Image.new('L',(dark.shape[1]+2,dark.shape[0]+2))
            shape.paste(Image.fromarray(dark),(1,1))
            ImageDraw.floodfill(shape,(0,0),128)
            filled = Image.fromarray(np.where(np.array(shape)[1:-1,1:-1]==128,0,255).astype(np.uint8))
            layer = Image.new('L',original.size)
            layer.paste(filled,(box[0],box[1]))
            mask = layer.resize((width,height),Image.Resampling.LANCZOS)
            self.mask = Image.fromarray(np.maximum(np.array(self.mask),np.array(mask)))
        self.desk = self.bg.copy()
        self.desk.putalpha(self.mask)
        font = Path('C:/Windows/Fonts/YuGothB.ttc')
        if not font.exists():
            font = Path('C:/Windows/Fonts/meiryo.ttc')
        self.font_path = str(font)
        self.font = ImageFont.truetype(str(font), round(width*.047))
        self.small = ImageFont.truetype(str(font), round(width*.023))
        self.label = ImageFont.truetype(str(font), round(width*.033))
        self.cache = {}

    def sprite(self, name, height):
        key = (name, height)
        if key not in self.cache:
            src = self.poses[name]
            self.cache[key] = src.resize((round(src.width*height/src.height),height), Image.Resampling.LANCZOS)
        return self.cache[key].copy()

    def world(self, scene, t, speaking=False):
        w,h = self.w,self.h
        frame = self.bg.copy()
        shadow = Image.new('RGBA',(w,h))
        draw = ImageDraw.Draw(shadow)
        x,y = scene['x']*w,scene['footY']*h
        body_h = round(scene['bodyHeight']*h)
        draw.ellipse((x-w*.09,y-h*.008,x+w*.09,y+h*.012),fill=(24,17,32,85))
        frame.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(w*.012)))
        pose = scene['pose']
        src = self.sprite(pose,body_h)
        src = mouth_only(src,float(speaking),self.story.get('mouths',{}).get(pose))
        # 靴を固定し、髪と上体だけに微小な横揺れを与える。
        drift = math.sin(t*1.7)*w*.002
        src = src.transform(src.size,Image.Transform.AFFINE,(1,drift/body_h,-drift,0,1,0),Image.Resampling.BICUBIC)
        if scene.get('behindDesk'):
            y += (1-ease((t-.3)/1.1))*h*scene.get('peekDrop',.035) + math.sin(t*2)*h*.001
        frame.alpha_composite(src,(round(x-src.width/2),round(y-src.height)))
        if scene.get('behindDesk'):
            frame.alpha_composite(self.desk)
        z = scene.get('camera',1)+.012*ease((t-scene.get('start',0))/5)
        cx,cy = scene.get('focus',[.5,.5])
        cw,ch = w/z,h/z
        left = max(0,min(w-cw,cx*w-cw/2))
        top = max(0,min(h-ch,cy*h-ch/2))
        return frame.crop((round(left),round(top),round(left+cw),round(top+ch))).resize((w,h),Image.Resampling.BICUBIC)

    def render(self, t, timeline, pcm, duration):
        scene = next((s for s in timeline if t < s['end']),timeline[-1])
        talking = mouth_at(scene,t)
        frame = self.scene_world(scene,t,talking)
        index = timeline.index(scene)
        # ポーズを単純な縦縮小で代用せず、専用画像へ短いディゾルブで接続。
        if index and scene.get('transition','dissolve') != 'cut' and t-scene['start'] < .20:
            previous = timeline[index-1]
            frame = Image.blend(self.scene_world(previous,min(t,previous['end']-.001)),frame,ease((t-scene['start'])/.20))
        overlay = Image.new('RGBA',frame.size)
        d = ImageDraw.Draw(overlay)
        w,h = self.w,self.h
        d.rounded_rectangle((w*.07,h*.055,w*.60,h*.102),radius=round(w*.02),fill=(25,32,51,200))
        d.text((w*.095,h*.067),'望月京子｜はじめまして',font=self.small,fill=(255,246,232,255))
        if t >= scene['start'] and t < scene['speechEnd']+.3:
            lines = scene['caption'].split('\n')
            widths = [d.textlength(line,font=self.font) for line in lines]
            box_width = max(widths)+w*.10
            top = h*.835
            d.rounded_rectangle(((w-box_width)/2,top,(w+box_width)/2,top+h*.095),radius=round(w*.022),fill=(23,30,49,229),outline=(222,202,173,170),width=max(1,round(w*.002)))
            for i,line in enumerate(lines):
                d.text(((w-widths[i])/2,top+h*.014+i*h*.033),line,font=self.font,fill=(255,249,238,255),stroke_width=max(0,round(w*.0008)),stroke_fill=(25,30,48,255))
        if index == 4 and scene.get('showNameLabel',True):
            d.text((w*.47,h*.18),'望月 京子',font=self.label,fill=(255,246,230,255),stroke_width=2,stroke_fill=(30,35,55,240))
        credit = 'VOICEVOX:ナースロボ＿タイプＴ'
        tw = d.textlength(credit,font=self.small)
        d.text(((w-tw)/2,h*.955),credit,font=self.small,fill=(246,239,232,255),stroke_width=1,stroke_fill=(20,24,40,255))
        frame.alpha_composite(overlay)
        fade = min(ease(t/.35),ease((duration-t)/.55))
        if fade < 1:
            frame = Image.blend(Image.new('RGBA',frame.size,(19,22,36,255)),frame,fade)
        return frame.convert('RGB')

    def scene_world(self, scene, t, speaking=False):
        # 場面内の動きを専用ポーズの時系列で記述し、同じ接地位置で接続する。
        sequence = scene.get('poseSequence')
        if not sequence:
            return self.world(scene,t,speaking)
        span = scene['end']-scene['start']
        elapsed = max(0,t-scene['start'])
        index = max(i for i,key in enumerate(sequence) if elapsed >= key['at']*span)
        current = {**scene,'pose':sequence[index]['pose']}
        frame = self.world(current,t,speaking)
        since = elapsed-sequence[index]['at']*span
        transition = scene.get('poseTransition',.14)
        if index and since < transition:
            previous = {**scene,'pose':sequence[index-1]['pose']}
            frame = Image.blend(self.world(previous,t,speaking),frame,ease(since/transition))
        return frame


def timestamp(seconds):
    ms = round(seconds*1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='projects/short-001/storyboard.json')
    parser.add_argument('--engine',default=os.environ.get('VOICEVOX_URL','http://127.0.0.1:50031'))
    parser.add_argument('--output',default=None)
    parser.add_argument('--preview',action='store_true',help='軽量な540×960 MP4を出力')
    parser.add_argument('--reuse-audio',action='store_true',help='同一入力の音声を保存済みmanifestから再利用')
    args = parser.parse_args()
    project = ROOT/args.project
    story = json.loads(project.read_text(encoding='utf-8'))
    voice = json.loads((ROOT/story['voiceConfig']).read_text(encoding='utf-8'))
    if not story['scenes'] or story['fps'] <= 0 or story['width']%2 or story['height']%2:
        raise ValueError('台本と出力形式を確認してください')
    out = ROOT/(args.output or f'outputs/shorts/{story["id"]}')
    out.mkdir(parents=True,exist_ok=True)
    # 画角や遮蔽の変更では音声を再合成しない。台詞・余韻・声だけで照合する。
    def audio_signature(scenes):
        return hashlib.sha256(json.dumps([[[s.get('speechText',s['text']),s['pauseAfter']] for s in scenes],voice],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    audio_key = audio_signature(story['scenes'])
    manifest_file = out/'manifest.json'
    if args.reuse_audio:
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        old_scenes = [{k:v for k,v in s.items() if k not in ['start','speechEnd','end','audio']} for s in manifest['timeline']]
        legacy_key = hashlib.sha256(json.dumps([old_scenes,voice],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        source_valid = manifest['audioInputHash'] in [audio_signature(old_scenes),legacy_key]
        if not source_valid or audio_signature(old_scenes) != audio_key or manifest['soundtrackHash'] != digest(out/'soundtrack.wav'):
            raise ValueError('音声入力または保存音声が変更されています。再生成してください')
        timeline = [{**old,**new} for old,new in zip(manifest['timeline'],story['scenes'])]
        duration,engine = manifest['duration'],manifest['engine']
        with wave.open(str(out/'soundtrack.wav')) as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()),'<i2').astype(float)/32768
    else:
        timeline,duration,pcm,engine = make_audio(story,voice,out,args.engine)
    width,height = (540,960) if args.preview else (story['width'],story['height'])
    stage = Stage(story,width,height)
    fps = story['fps']
    count = math.ceil(duration*fps)
    movie = out/('preview.mp4' if args.preview else 'kyoko-short-001.mp4')
    temporary = out/(movie.stem+'.partial.mp4')
    manifest = {'project':args.project,'title':story['title'],'audioInputHash':audio_key,'soundtrackHash':digest(out/'soundtrack.wav'),'duration':duration,'fps':fps,'width':width,'height':height,'timeline':timeline,'engine':engine,'credit':voice['credit'],'assets':{p:digest(ROOT/p) for p in [story['background'],*story['poses'].values()]},'rendererHash':digest(Path(__file__)),'storyboardHash':digest(project),'output':str(movie.relative_to(ROOT)),'status':'書き出し前'}
    manifest['rendererDependencies'] = {'tools/shorts/lipsync.py':digest(Path(__file__).with_name('lipsync.py'))}
    manifest_file.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    command = [find_binary('ffmpeg'),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-i',str(out/'soundtrack.wav'),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-af','loudnorm=I=-16:TP=-1.5:LRA=11','-movflags','+faststart','-shortest',str(temporary)]
    process = subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for i in range(count):
            process.stdin.write(stage.render(i/fps,timeline,pcm,duration).tobytes())
            if i%150 == 0:
                print(f'描画 {i}/{count}',flush=True)
        process.stdin.close()
        error = process.stderr.read().decode(errors='replace')
        if process.wait():
            raise RuntimeError(error)
    except BaseException as exception:
        process.kill()
        process.wait()
        error = process.stderr.read().decode(errors='replace')
        raise RuntimeError(f'エンコーダー停止（終了値 {process.returncode}）: {error}') from exception
    temporary.replace(movie)
    manifest['status'] = '書き出し完了'
    manifest['outputHash'] = digest(movie)
    manifest_file.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    subtitles = '\n\n'.join(f'{i+1}\n{timestamp(s["start"])} --> {timestamp(s["speechEnd"])}\n{s["caption"]}' for i,s in enumerate(timeline))
    (out/'subtitles.srt').write_text(subtitles,encoding='utf-8')
    (out/'CREDITS.txt').write_text(voice['credit']+'\n画像: OpenAI image_gen（既存キャラクター原画を参照）\nベル音: 本プロジェクトで数式合成\n',encoding='utf-8')
    thumbs = []
    for i,s in enumerate(timeline):
        img = stage.render(s['start']+.8,timeline,pcm,duration)
        img.save(out/f'scene-{i+1:02}.png')
        thumbs.append(img.resize((270,480)))
    contact = Image.new('RGB',(270*len(thumbs),480))
    for i,img in enumerate(thumbs):
        contact.paste(img,(i*270,0))
    contact.save(out/'contact-sheet.jpg')
    page = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>京子の最初のショート</title><style>body{background:#171c2b;color:#faf1e4;font-family:system-ui;text-align:center}video{height:85vh;max-width:95vw}a{color:#e9cba9}</style><h1>隠れてご挨拶</h1><video controls src="'+movie.name+'"></video><p>VOICEVOX:ナースロボ＿タイプＴ</p><p><a href="'+movie.name+'" download>MP4を保存</a></p></html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_visual_review.py')),'--output',str(out)],check=True)
    print(f'書き出し完了・視覚レビュー未完了: {movie} ({duration:.2f}秒)',flush=True)


if __name__ == '__main__':
    main()
