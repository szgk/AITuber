"""VOICEVOXの音素長を音声実尺へ合わせ、口元だけを動かす。"""
import numpy as np
import math
from PIL import Image,ImageDraw


def phoneme_cues(query,duration,start=0):
    segments = [(query['prePhonemeLength'], 'pau')]
    for phrase in query['accent_phrases']:
        for mora in phrase['moras']:
            if mora.get('consonant') and mora.get('consonant_length'):
                segments.append((mora['consonant_length'],mora['consonant']))
            segments.append((mora['vowel_length'],mora['vowel']))
        pause = phrase.get('pause_mora')
        if pause:
            segments.append((pause['vowel_length'],'pau'))
    segments.append((query['postPhonemeLength'],'pau'))
    factor = duration/sum(length for length,_ in segments)
    cursor = start
    cues = []
    for length,phone in segments:
        end = cursor+length*factor
        openness = {'pau':0,'sil':0,'cl':0,'a':1,'i':.5,'u':.55,'e':.8,'o':.85,'N':.25,'m':.12,'b':.12,'p':.12}.get(phone,.38)
        cues.append({'start':cursor,'end':end,'phoneme':phone,'open':openness})
        cursor = end
    return cues


def mouth_at(scene,t):
    for cue in scene.get('mouthCues',[]):
        if cue['start'] <= t < cue['end']:
            return cue['open']
    return 0


def mouth_only(image,openness,spec):
    if not spec or openness <= .01:
        return image.copy()
    result = image.copy()
    w,h = image.size
    left,top,right,bottom = [round(v*(w if i%2==0 else h)) for i,v in enumerate(spec['erase'])]
    pixels = np.array(image)
    # 既存の閉じ口だけを上下の肌色から補間。目や頬を含む顔差分は使わない。
    for y in range(top,bottom):
        alpha = (y-top+1)/(bottom-top+1)
        pixels[y,left:right] = np.round((1-alpha)*pixels[top-1,left:right]+alpha*pixels[bottom,left:right]).astype(np.uint8)
    result = Image.fromarray(pixels,'RGBA')
    cx,cy = spec['center'][0]*w,spec['center'][1]*h
    rx,ry = spec.get('width',.055)*w/2,max(1,spec.get('height',.014)*h*openness/2)
    # 口だけを高倍率で描き、面積平均に近い縮小で輪郭の段差を除く。
    scale = 8
    patch = result.crop((left,top,right,bottom)).resize(((right-left)*scale,(bottom-top)*scale),Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(patch)
    # 画像座標の角度で回転する。負角は右の口角が上がる向き。
    angle = math.radians(spec.get('angleDegrees',0))
    cosine,sine = math.cos(angle),math.sin(angle)
    def ellipse(radius_x,radius_y,offset_y,color):
        points=[]
        for step in range(128):
            phase=step*2*math.pi/128
            u,v=radius_x*math.cos(phase),offset_y+radius_y*math.sin(phase)
            x,y=cx+u*cosine-v*sine,cy+u*sine+v*cosine
            points.append(((x-left)*scale,(y-top)*scale))
        draw.polygon(points,fill=color)
    stroke=max(1/scale,h*.0015)
    ellipse(rx,ry,0,(29,26,31,255))
    ellipse(max(.1,rx-stroke),max(.1,ry-stroke),0,(94,38,45,255))
    if openness > .45:
        ellipse(rx*.65,ry*.3,ry*.4,(222,130,139,255))
    result.paste(patch.resize((right-left,bottom-top),Image.Resampling.LANCZOS),(left,top))
    return result
