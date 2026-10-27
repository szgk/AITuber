"""口角軸と描画角の一致を検査し、顔の動きとの合成順も検証する。"""
import math
import unittest
import numpy as np
from debut_animation import mouth_layer,MOUTH_ANGLES,Rig,S

class MouthAngleTest(unittest.TestCase):
 def test_open_mouth_follows_original_tilt(self):
  for n,expected in MOUTH_ANGLES.items():
   with self.subTest(panel=n):
    alpha=np.array(mouth_layer(n,.55).getchannel('A'))
    y,x=np.where(alpha>160)
    eig,vec=np.linalg.eigh(np.cov(np.stack((x,y))))
    axis=vec[:,np.argmax(eig)]
    actual=math.degrees(math.atan2(axis[1],axis[0]))
    actual=(actual+90)%180-90
    self.assertAlmostEqual(actual,expected,delta=2)
 def test_mouth_openness_does_not_change_eyes(self):
  for n in (9,14,15):
   r=Rig(n)
   a=np.array(r.frame(.4,0));b=np.array(r.frame(.4,.8))
   self.assertTrue(np.array_equal(a[:260,:],b[:260,:]))

if __name__=='__main__':unittest.main()
