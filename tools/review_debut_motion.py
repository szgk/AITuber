"""実MP4から全編の連続フレームと顔・口・毛先の証拠を抽出する。"""
from pathlib import Path
import subprocess,sys,json,hashlib,argparse
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parent/'shorts'))
from render import find_binary
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
root=args.output.resolve() if args.output else Path(__file__).resolve().parents[1]/'outputs/debut-motion-preview'
ff=find_binary('ffmpeg')
for video in root.glob('*/video.mp4'):
 if video.stat().st_size==0:continue
 out=video.parent/'review';out.mkdir(exist_ok=True)
 subprocess.run([ff,'-hide_banner','-loglevel','error','-y','-i',str(video),'-vf','fps=10,scale=360:360',str(out/'frame-%03d.png')],check=True)
 files=sorted(out.glob('frame-*.png'))
 sheet=Image.new('RGB',(1800,1080),'white')
 for i,p in enumerate(files):sheet.paste(Image.open(p).resize((180,180)),((i%10)*180,(i//10)*180))
 sheet.save(out/'full-sequence.png')
 subprocess.run([ff,'-hide_banner','-loglevel','error','-y','-ss','0.5','-i',str(video),'-frames:v','7','-vf','crop=1080:890:0:60,scale=540:445',str(out/'detail-%02d.png')],check=True)
 detail=Image.new('RGB',(1620,1335),'white')
 for i,p in enumerate(sorted(out.glob('detail-*.png'))):
  im=Image.open(p)
  detail.paste(im,((i%3)*540,(i//3)*445))
 # 7枚を1行に縮小せず、実際の原寸詳細を別ファイルにも保持する。
 detail.save(out/'face-sequence.png')
 print(video.parent.name, len(files),flush=True)
