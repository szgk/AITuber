"""原画の手の肌領域と輪郭を抽出し、剛体回転用レイヤーを作る。"""
from pathlib import Path
from collections import deque
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFilter

root=Path(__file__).resolve().parents[1]/'assets/debut-animation'
out=root/'rigid-hands'
settings={3:{'seed':[91,344],'bounds':[0,195,185,400],'pivot':[93,392],'angle':4},15:{'seed':[156,460],'bounds':[25,290,285,568],'pivot':[157,550],'angle':6}}
for n,spec in settings.items():
 src=Image.open(root/f'{n:02}.png').convert('RGB');a=np.array(src)
 valid=(a[:,:,0]>170)&(a[:,:,1]>120)&(a[:,:,2]>105)
 l,t,r,b=spec['bounds'];valid[:t]=False;valid[b:]=False;valid[:,:l]=False;valid[:,r:]=False
 mask=np.zeros(valid.shape,dtype=np.uint8);q=deque([tuple(spec['seed'])])
 while q:
  x,y=q.popleft()
  if not(0<=x<600 and 0<=y<600) or not valid[y,x] or mask[y,x]:continue
  mask[y,x]=255;q.extend([(x-1,y),(x+1,y),(x,y-1),(x,y+1)])
 assert mask.sum()>1000*255
 alpha=Image.fromarray(mask).filter(ImageFilter.MaxFilter(9))
 # 手首の切断面は白い袖口の内側に置く。
 cutoff=Image.new('L',(600,600));ImageDraw.Draw(cutoff).rectangle((0,0,600,397 if n==3 else 565),fill=255)
 alpha=Image.fromarray(np.minimum(np.array(alpha),np.array(cutoff))).filter(ImageFilter.GaussianBlur(.35))
 layer=src.convert('RGBA');layer.putalpha(alpha);layer.save(out/f'{n:02}-hand.png')
 clean=Image.open(out/f'{n:02}-clean-generated.png').convert('RGB').resize((600,600),Image.Resampling.LANCZOS)
 # 補完画像を使うのは元の手の後ろだけ。顔・髪・衣装の他の画素は元絵を保持。
 erase=alpha.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(.8))
 if n==3:
  # 灰色背景の補完差を手の形の縁として残さないよう矩形の下絵へ接続する。
  erase=Image.new('L',(600,600));ImageDraw.Draw(erase).rectangle((0,194,185,402),fill=255)
  erase=erase.filter(ImageFilter.GaussianBlur(3))
 plate=src.copy();plate.paste(clean,(0,0),erase);plate.save(out/f'{n:02}-plate.png')
 spec['hand']=f'{n:02}-hand.png';spec['plate']=f'{n:02}-plate.png'
(out/'rigs.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
print('03・15の手と補完下絵を保存')
