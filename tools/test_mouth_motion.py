"""口の開閉の連続性、無音時の閉口、時刻の維持を検査する。"""
import unittest
import numpy as np
from mouth_motion import mouth_keyframes,smooth_mouth_at

class MouthMotionTest(unittest.TestCase):
 def setUp(self):
  self.cues=[{'start':0.,'end':.4,'open':0.},{'start':.4,'end':.6,'open':1.},{'start':.6,'end':.8,'open':.35},{'start':.8,'end':1.2,'open':0.}]
  self.keys=mouth_keyframes({'mouthCues':self.cues})
 def test_zero_during_silence(self):
  for t in list(np.linspace(0,.4,30))+list(np.linspace(.8,1.2,30)):
   self.assertAlmostEqual(smooth_mouth_at(self.keys,float(t)),0.)
 def test_continuous_and_no_overshoot(self):
  values=[smooth_mouth_at(self.keys,float(t)) for t in np.linspace(0,1.2,12001)]
  self.assertGreaterEqual(min(values),0);self.assertLessEqual(max(values),1)
  self.assertLess(max(abs(np.diff(values))),.005)
  self.assertGreater(len(set(round(v,4) for v in values)),100)
 def test_targets_and_timing_are_retained(self):
  self.assertAlmostEqual(smooth_mouth_at(self.keys,.5),1.)
  self.assertAlmostEqual(smooth_mouth_at(self.keys,.7),.35)
  self.assertAlmostEqual(smooth_mouth_at(self.keys,.4),0.)
  self.assertAlmostEqual(smooth_mouth_at(self.keys,.8),0.)

if __name__=='__main__':unittest.main()
