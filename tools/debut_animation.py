"""告知用ラフを音声・局所メッシュ・独立した目の演技で動画化する。"""
from pathlib import Path
import argparse, hashlib, json, math, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/shorts'))
from render import make_audio, find_binary
from lipsync import mouth_at
from mouth_motion import mouth_keyframes,smooth_mouth_at
ASSETS=ROOT/'assets/debut-animation'
OUT=ROOT/'outputs/debut-videos-v1'
FPS=30
S=600
TEXTS=[
'あ、あの……はじめまして。望月京子です。……頭しか出てませんでした。',
'昨日より、少しだけ。ご挨拶できて、ちょっとうれしいです。',
'こんにちは……。手のほうが先に、ご挨拶できました。',
'小さくて、かわいいものが好きです。……つい、近くで見たくなります。',
'好きなお寿司は、サーモンです。ここは、迷いません。',
'ご挨拶、少し慣れてきました。……今、笑ったところも見えてました？',
'ゲームは、RPGとリズムゲームが好きです。……好きな話だと、長くなりますね。',
'あと七日で、デビューです。今日は、片目だけ……。目が合うと、緊張しますね。',
'あと六日……。こんばんは、望月京子です。……今日は、隠れずに言えました。',
'あと五日。大きな犬が好きです。あの重たさも、好きなんです。',
'あと四日。お洋服も、少しだけ。……これで、大丈夫かな。',
'あと三日。のんびり、おはなしできたら、うれしいです。',
'あと二日。はじめまして……こんにちは。……どちらから言おう。',
'明日、デビューです。緊張してます。でも……楽しみも、増えました。',
'はじめまして、望月京子です。あ、あの……これから、よろしくお願いします！']
MOUTHS={6:(.53,.263,.15,.075),7:(.527,.238,.235,.20),8:(.29,.68,.075,.035),9:(.56,.63,.20,.085),10:(.605,.642,.185,.16),11:(.574,.518,.125,.065),12:(.571,.534,.125,.065),13:(.568,.544,.08,.07),14:(.55,.55,.15,.13),15:(.59,.55,.22,.16)}
# 原画の左右の口角を結ぶ軸。画像座標では右上がりを負角にする。
MOUTH_ANGLES={6:-17,7:-14,8:-14,9:-16,10:-15,11:-18,12:-17,13:-10,14:-12,15:-16}
SPIRALS={9:[(.353,.463,.067),(.718,.410,.062)],14:[(.351,.431,.064),(.703,.385,.061)]}
HIGHLIGHTS={8:[(.442,.39,.036)],11:[(.39,.43,.028),(.735,.365,.027)],13:[(.40,.45,.026),(.75,.35,.025)]}
HAND_DIR=ASSETS/'rigid-hands'
# 小道具や頬へ添えた手は保持。手へのガウス変形は行わない。
FIXED_HAND_REGIONS={4:(.58,.09,1,.79),5:(.50,.12,1,.83),11:(.17,.67,.55,1),13:(.22,.58,.52,1),14:(.26,.69,.81,1)}

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ease(x):
 x=max(0,min(1,x));return x*x*(3-2*x)

def mouth_layer(n,opening):
 """口の局所座標で描画し、原画の口角軸へ回転してから合成する。"""
 x,y,w,h=MOUTHS[n];x*=S;y*=S;w*=S;h*=S
 k=3;size=240
 patch=Image.new('RGBA',(size*k,size*k));d=ImageDraw.Draw(patch)
 cx=cy=size*k/2
 opening=max(0.,min(1.,opening))
 upper=[];lower=[]
 for u in np.linspace(-1,1,97):
  closed=h*.10*(math.sin(u*math.pi*3) if n==9 else 1-u*u)
  center=closed*(1-opening)
  aperture=h*.48*opening*math.sqrt(max(0.,1-u*u))
  x=cx+w*(.30+.17*opening)*u*k
  upper.append((x,cy+(center-aperture)*k))
  lower.append((x,cy+(center+aperture)*k))
 if opening>.001:
  contour=upper+list(reversed(lower))
  d.polygon(contour,fill='#db8793')
  d.line(contour+[contour[0]],fill='#211e22',width=3*k,joint='curve')
 else:d.line(upper,fill='#211e22',width=3*k,joint='curve')
 # PILは反時計回りが正。画面座標の軸角と符号が逆になる。
 patch=patch.rotate(-MOUTH_ANGLES[n],Image.Resampling.BICUBIC).resize((size,size),Image.Resampling.LANCZOS)
 layer=Image.new('RGBA',(S,S));layer.alpha_composite(patch,(round(x-size/2),round(y-size/2)))
 return layer

class Rig:
 def __init__(self,n):
  self.n=n;self.base=Image.open(ASSETS/f'{n:02}.png').convert('RGB')
  self.original=self.base.copy();self.hand=None;self.hand_spec=None
  if n in (3,15):
   self.hand_spec=json.loads((HAND_DIR/'rigs.json').read_text(encoding='utf-8'))[str(n)]
   self.hand=Image.open(HAND_DIR/self.hand_spec['hand']).convert('RGBA')
   self.base=Image.open(HAND_DIR/self.hand_spec['plate']).convert('RGB')
  self.closed=self.base.copy()
  if n in MOUTHS:
   x,y,w,h=MOUTHS[n];x,y,w,h=[v*S for v in (x,y,w,h)]
   l,r=int(x-w/2-4),int(x+w/2+4);a,b=int(y-h/2-7),int(y+h/2+7)
   arr=np.array(self.base)
   color=tuple(int(v) for v in np.median(arr[max(0,a-5):a,l:r].reshape(-1,3),axis=0))
   mask=Image.new('L',(S,S));md=ImageDraw.Draw(mask)
   if n==15:
    md.polygon([(282,282),(408,264),(432,317),(405,370),(365,392),(304,374),(279,332)],fill=255)
   else:md.rounded_rectangle((l,a,r,b),radius=8,fill=255)
   self.closed.paste(Image.new('RGB',(S,S),color),(0,0),mask.filter(ImageFilter.GaussianBlur(2)))
  self.nodes=[]
  for y in range(0,S,20):
   for x in range(0,S,20): self.nodes.append((x,y,x+20,y+20))

 def features(self,t,opening):
  im=self.closed.copy();d=ImageDraw.Draw(im)
  if self.n in MOUTHS:
   layer=mouth_layer(self.n,opening)
   im.paste(layer,(0,0),layer)
   d=ImageDraw.Draw(im)
  for j,(cx,cy,r) in enumerate(SPIRALS.get(self.n,[])):
   cx*=S;cy*=S;r*=S
   # 回転するのは白目内の渦巻きのみ。顔やまぶたは回さない。
   d.ellipse((cx-r,cy-r,cx+r,cy+r),fill='white')
   phase=t*2.8*(1 if j==0 else -1)
   points=[]
   for a in np.linspace(0,math.pi*4.5,160):
    rr=r*.87*a/(math.pi*4.5)
    points.append((cx+rr*math.cos(a+phase),cy+rr*math.sin(a+phase)))
   d.line(points,fill='#17171b',width=6)
  # 目元のうるうるは発話と独立したゆっくりした周期。
  for cx,cy,r in HIGHLIGHTS.get(self.n,[]):
   cx=cx*S+1.6*math.sin(t*4);cy=cy*S+1.2*math.cos(t*4.4);r*=S*.16
   d.ellipse((cx-r,cy-r,cx+r,cy+r),fill='white')
   d.ellipse((cx+r*1.5,cy+r,cx+r*2.4,cy+r*1.9),fill='#f7f7f7')
  return im

 def source_xy(self,x,y,t):
  u,v=x/S,y/S;n=self.n
  # 境界を固定し、頭〜顔を中心に左右と上下へ滑らかに動かす。
  border=min(1,u/.08,(1-u)/.08,v/.055,(1-v)/.08);border=max(0,border)
  head=max(0,1-(max(0,v-.70)/.30))
  sx=2.2*math.sin(t*1.7)*head;sy=1.5*math.sin(t*2.1)*head
  # 覗き込みは目隠しの右へ顔を寄せ、少し前に出て戻る動作。
  if n==8:
   reveal=math.sin(math.pi*ease((t%5)/5))
   sx+=-10*reveal*max(0,min(1,(u-.28)/.16));sy+=-3*reveal
  if n<=5:
   # 画面下の目隠しを固定し、頭だけを持ち上げる。
   band=max(0,min(1,(.70-v)/.12))
   sx*=band;sy=(5*math.sin(t*1.7)+9*(1-ease(t/1.0)))*band
  # 向かって右上の跳ね毛。付け根を固定し毛先ほど振幅を増やす。
  curl_y=.38 if n<=5 else .17
  region=math.exp(-(((u-.88)/.17)**2+((v-curl_y)/.18)**2)*2)
  tip=max(0,min(1,(u-.68)/.26))
  happy=n in (4,5,7,10,12,14,15)
  amplitude=15 if happy else 9
  wag=math.sin(t*(6.2 if happy else 3.8))+.22*math.sin(t*10.1)
  sy+=amplitude*wag*region*tip
  sx+=amplitude*.35*math.cos(t*(6.2 if happy else 3.8))*region*tip
  if n in FIXED_HAND_REGIONS:
   l,a,r,b=FIXED_HAND_REGIONS[n]
   if l<=u<=r and a<=v<=b:
    # 手の内部は剛体の恒等変換に固定し、外周だけ胴体へ緩く接続する。
    keep=max(0,min(1,(u-l)/.025,(r-u)/.025,(v-a)/.025,(b-v)/.025))
    sx*=1-keep;sy*=1-keep
  return x+sx*border,y+sy*border

 def hand_transform(self,t):
  """手全体へ同一の回転と並進だけを適用。縮尺・せん断・局所変形なし。"""
  spec=self.hand_spec;px,py=spec['pivot']
  angle=spec['angle']*math.sin(t*4.4)
  # 付け根の位置は下絵の身体の微小移動と共有する。
  sx,sy=self.source_xy(px,py,t);dx,dy=px-sx,py-sy
  a=math.radians(angle);c,s=math.cos(a),math.sin(a)
  return (c,-s,px+dx-c*px+s*py,s,c,py+dy-s*px-c*py)

 def rigid_hand(self,t):
  a,b,c,d,e,f=self.hand_transform(t)
  # 順変換をPillowの逆写像へ変換。回転行列の逆は転置。
  inverse=(a,d,-a*c-d*f,b,e,-b*c-e*f)
  return self.hand.transform((S,S),Image.Transform.AFFINE,inverse,Image.Resampling.BICUBIC)

 def frame(self,t,op):
  im=self.features(t,op);mesh=[]
  for box in self.nodes:
   l,a,r,b=box
   q=[]
   for x,y in [(l,a),(l,b),(r,b),(r,a)]:q.extend(self.source_xy(x,y,t))
   mesh.append((box,tuple(q)))
  result=im.transform((S,S),Image.Transform.MESH,mesh,Image.Resampling.BICUBIC)
  if self.hand is not None:
   hand=self.rigid_hand(t);result.paste(hand,(0,0),hand)
   if self.n==3:
    # 袖口の前側を手首へ重ね、回転時にも接続面を隠す。
    mask=Image.new('L',(S,S));ImageDraw.Draw(mask).polygon([(58,394),(129,394),(130,414),(57,414)],fill=255)
    result.paste(self.original,(0,0),mask)
  return result

def caption_lines(text,limit=21):
 lines=[]
 for piece in text.replace('。','。\n').splitlines():
  while len(piece)>limit:lines.append(piece[:limit]);piece=piece[limit:]
  if piece:lines.append(piece)
 return lines

def render(n,reuse=False,motion_preview=False):
 out=OUT/f'{n:02}-d-{15-n:02}';out.mkdir(parents=True,exist_ok=True)
 voice=json.loads((ROOT/'config/kyoko-voice.json').read_text(encoding='utf-8'))
 story={'scenes':[{'text':TEXTS[n-1],'speechText':TEXTS[n-1],'pauseAfter':.75}]}
 if motion_preview:
  duration=6.0
  timeline=[{'mouthCues':[{'start':.5+j*.22,'end':.5+(j+1)*.22,'open':(0,.55,.9,.35)[j%4]} for j in range(20)]}]
  with wave.open(str(out/'soundtrack.wav'),'wb') as wav:
   wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(bytes(2*24000*6))
 elif reuse and (out/'timeline.json').exists():
  meta=json.loads((out/'timeline.json').read_text(encoding='utf-8'));timeline=meta['timeline'];duration=meta['duration']
 else:
  timeline,duration,_,engine=make_audio(story,voice,out,'http://127.0.0.1:50031')
  (out/'timeline.json').write_text(json.dumps({'timeline':timeline,'duration':duration,'voice':voice,'engine':engine},ensure_ascii=False,indent=2),encoding='utf-8')
 rig=Rig(n);font=ImageFont.truetype('C:/Windows/Fonts/YuGothB.ttc',24);small=ImageFont.truetype('C:/Windows/Fonts/YuGothB.ttc',20)
 mouth_keys=mouth_keyframes(timeline[0])
 (out/'mouth-motion.json').write_text(json.dumps({'transitionSeconds':.055,'interpolation':'5次補間（C2連続）','keyframes':mouth_keys,'rawCues':timeline[0]['mouthCues'],'silentMotionPreview':motion_preview},ensure_ascii=False,indent=2),encoding='utf-8')
 ff=find_binary('ffmpeg');video=out/'video.mp4'
 cmd=[ff,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','720x720','-r',str(FPS),'-i','pipe:0','-i',str(out/'soundtrack.wav'),'-vf','scale=1080:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-af','loudnorm=I=-16:TP=-1.5:LRA=9','-c:a','aac','-b:a','160k','-t',str(duration),'-movflags','+faststart',str(video)]
 process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 count=math.ceil(duration*FPS)
 for f in range(count):
  t=f/FPS;im=Image.new('RGB',(720,720),'#f6f6f2')
  art=rig.frame(t,smooth_mouth_at(mouth_keys,t)).resize((720,590),Image.Resampling.BICUBIC)
  im.paste(art,(0,40));d=ImageDraw.Draw(im)
  label='動き確認用・無音／口はテスト用の開閉' if motion_preview else f'デビューまであと{15-n}日' if n<15 else '望月京子  はじめまして'
  d.rectangle((0,0,720,40),fill='#233449');d.text((22,5),label,font=small,fill='white')
  lines=caption_lines(TEXTS[n-1]);seg=max(0,min(len(lines)-1,int((t-.55)/max(.1,(duration-1.3))*len(lines))))
  d.text((24,647),'顔・口・目・手・くせ毛の動作確認' if motion_preview else lines[seg],font=font,fill='#233449')
  process.stdin.write(im.tobytes())
 process.stdin.close();rc=process.wait()
 if rc:raise RuntimeError(f'FFmpeg失敗: {n}')
 (out/'字幕.txt').write_text(TEXTS[n-1],encoding='utf-8')
 (out/'credit.txt').write_text(voice['credit'],encoding='utf-8')
 (out/'manifest.json').write_text(json.dumps({'video':video.name,'sha256':digest(video),'asset':f'assets/debut-animation/{n:02}.png','assetSha256':digest(ASSETS/f'{n:02}.png'),'rendererSha256':digest(Path(__file__)),'duration':duration,'fps':FPS,'size':[1080,1080],'status':'視覚レビュー待ち','motion':['頭と顔の揺れ','右上の跳ね毛の局所変形']+(['手振り'] if n in (3,15) else [])+(['覗き込み'] if n==8 else [])+(['ぐるぐる目の回転'] if n in SPIRALS else [])+(['ハイライトの揺れ'] if n in HIGHLIGHTS else [])},ensure_ascii=False,indent=2),encoding='utf-8')
 if n in (3,15) or n in FIXED_HAND_REGIONS:
  manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
  manifest['handAnimation']='手首支点の剛体回転・並進' if n in (3,15) else '保持する手の内部を固定・局所変形なし'
  if n in (3,15):
   manifest['handInputs']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in [HAND_DIR/f'{n:02}-hand.png',HAND_DIR/f'{n:02}-plate.png',HAND_DIR/'rigs.json']]
  (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 print(f'動画 {n:02}: {duration:.2f}秒 -> {video}',flush=True)

def main():
 global OUT
 ap=argparse.ArgumentParser();ap.add_argument('--only',type=int,nargs='*');ap.add_argument('--reuse-audio',action='store_true');ap.add_argument('--motion-preview',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
 if args.motion_preview:OUT=ROOT/'outputs/debut-motion-preview'
 if args.output:OUT=args.output.resolve()
 for n in args.only or range(1,16):render(n,args.reuse_audio,args.motion_preview)
 entries=[]
 for p in sorted(OUT.glob('*/video.mp4')):
  entries.append(f'<article><h2>{p.parent.name}</h2><video controls preload="metadata" src="{p.relative_to(OUT).as_posix()}"></video></article>')
 (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>望月京子 告知動画</title><style>body{background:#eef0f2;color:#233449;font-family:sans-serif;margin:30px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}video{width:100%}h2{font-size:18px}</style><h1>望月京子 告知動画 v1</h1><p>VOICEVOX:ナースロボ＿タイプＴ ／ 日程案・公開日未定</p><main>'+''.join(entries)+'</main>',encoding='utf-8')
if __name__=='__main__':main()
