"""剛体の手振りで指間距離・面積・支点が保たれることを検証する。"""
import math,unittest
import numpy as np
from debut_animation import Rig,S,FIXED_HAND_REGIONS

class HandRigTest(unittest.TestCase):
 def test_rigid_transform_preserves_distances_and_area(self):
  points=np.array([[61,230],[130,220],[40,300],[90,345],[95,390]],dtype=float)
  for n in (3,15):
   rig=Rig(n)
   for t in np.linspace(0,6,61):
    a,b,c,d,e,f=rig.hand_transform(t);m=np.array([[a,b],[d,e]])
    moved=points@m.T+[c,f]
    self.assertAlmostEqual(np.linalg.det(m),1,places=12)
    for i in range(len(points)):
     self.assertAlmostEqual(np.linalg.norm(moved[i]-moved[0]),np.linalg.norm(points[i]-points[0]),places=10)
 def test_wrist_follows_body_without_stretch(self):
  for n in (3,15):
   rig=Rig(n);px,py=rig.hand_spec['pivot']
   for t in (.0,.2,.6,1.1):
    a,b,c,d,e,f=rig.hand_transform(t);sx,sy=rig.source_xy(px,py,t)
    self.assertAlmostEqual(a*px+b*py+c,2*px-sx,places=10)
    self.assertAlmostEqual(d*px+e*py+f,2*py-sy,places=10)
 def test_holding_hands_do_not_deform(self):
  for n,(l,a,r,b) in FIXED_HAND_REGIONS.items():
   rig=Rig(n)
   for t in (0,.5,1):
    for u,v in [((l+r)/2,(a+b)/2),(l+.05,a+.05)]:
     x,y=u*S,v*S;sx,sy=rig.source_xy(x,y,t)
     self.assertAlmostEqual(x,sx);self.assertAlmostEqual(y,sy)

if __name__=='__main__':unittest.main()
