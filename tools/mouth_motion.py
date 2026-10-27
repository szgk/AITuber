"""音素時刻を保ち、口の開き具合を連続曲線へ補間する。"""
from bisect import bisect_right

def mouth_keyframes(scene,transition=.055):
 cues=scene.get('mouthCues',[]);points=[]
 for i,cue in enumerate(cues):
  start,end=float(cue['start']),float(cue['end']);op=float(cue['open'])
  if end<=start:continue
  if op<=0:
   points.extend([(start,0.),(end,0.)]);continue
  prev=cues[i-1] if i else None;after=cues[i+1] if i+1<len(cues) else None
  pad=min(transition,(end-start)*.35)
  if prev is None or prev['open']<=0 or prev['end']<start:points.append((start,0.))
  points.extend([(start+pad,op),(end-pad,op)])
  if after is None or after['open']<=0 or after['start']>end:points.append((end,0.))
 # 同じ時刻の休止点は0に統一し、無音区間の中へ開口を漏らさない。
 result={}
 for t,op in points:result[t]=min(result.get(t,op),op)
 return sorted(result.items())

def smooth_mouth_at(keys,t):
 if not keys or t<keys[0][0] or t>=keys[-1][0]:return 0.
 index=bisect_right(keys,(t,float('inf')))-1
 a,oa=keys[index];b,ob=keys[index+1]
 u=max(0.,min(1.,(t-a)/(b-a)))
 blend=u*u*u*(u*(u*6-15)+10)
 return max(0.,min(1.,oa+(ob-oa)*blend))
