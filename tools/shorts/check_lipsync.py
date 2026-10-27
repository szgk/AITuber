"""実WAVの有音フレームと口の時刻、読みのモーラ列を確認する。"""
import argparse
import json
import wave
from pathlib import Path
import numpy as np
from render import ROOT
from lipsync import mouth_at


def check(folder):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    results=[]
    for scene in manifest['timeline']:
        with wave.open(str(ROOT/scene['audio'])) as wav:
            rate=wav.getframerate()
            pcm=np.frombuffer(wav.readframes(wav.getnframes()),'<i2').astype(float)/32768
        audible=opened=0
        for offset in np.arange(0,len(pcm)/rate,1/manifest['fps']):
            samples=pcm[int(offset*rate):int((offset+1/manifest['fps'])*rate)]
            if len(samples) and np.sqrt(np.mean(samples*samples))>.012:
                audible+=1
                opened+=mouth_at(scene,scene['start']+offset)>0
        query=json.loads((ROOT/scene['query']).read_text(encoding='utf-8'))
        kana=''.join(mora['text'] for phrase in query['accent_phrases'] for mora in phrase['moras'])
        coverage=opened/max(1,audible)
        assert coverage>.75,f'音声と口の時間対応が不足: {scene["text"]} {coverage}'
        for expected in scene.get('expectedKanaContains',[]):
            assert expected in kana,f'読みが異なります: {expected} / {kana}'
        results.append({'text':scene['text'],'speechText':scene.get('speechText',scene['text']),'kana':kana,'audibleFrames':audible,'mouthOpenFrames':int(opened),'coverage':round(float(coverage),3)})
    (folder/'lipsync-check.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/shorts/short-001-v4')
    args=parser.parse_args()
    print(json.dumps(check(ROOT/args.output),ensure_ascii=False,indent=2))
