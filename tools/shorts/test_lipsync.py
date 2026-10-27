"""口以外の不変性、全可視ポーズ、音素・無音区間の回帰検証。"""
import json
import unittest
import math
import numpy as np
from PIL import Image
from render import ROOT
from lipsync import mouth_only,phoneme_cues,mouth_at


class LipsyncTest(unittest.TestCase):
    def test_open_mouth_axis_matches_original_face_pose(self):
        story=json.loads((ROOT/'projects/short-001/storyboard.json').read_text(encoding='utf-8'))
        for pose,spec in story['mouths'].items():
            with self.subTest(pose=pose):
                (ax,ay),(bx,by)=spec['mouthAxis']
                expected=math.degrees(math.atan2(by-ay,bx-ax))
                self.assertAlmostEqual(expected,spec['angleDegrees'])
                image=Image.open(ROOT/story['poses'][pose]).convert('RGBA')
                image=image.crop(image.getchannel('A').point(lambda p:255 if p>24 else 0).getbbox())
                for openness in [.5,1]:
                    result=np.array(mouth_only(image,openness,spec))
                    x0,y0,x1,y1=[round(v*(image.width if i%2==0 else image.height)) for i,v in enumerate(spec['erase'])]
                    mask=result[y0:y1,x0:x1,:3].max(axis=2)<150
                    ys,xs=np.where(mask)
                    covariance=np.cov(np.stack([xs,ys]))
                    _,vectors=np.linalg.eigh(covariance);axis=vectors[:,-1]
                    if axis[0]<0:axis=-axis
                    actual=math.degrees(math.atan2(axis[1],axis[0]))
                    self.assertLess(abs(actual-expected),3,'開口の軸も原画の顔の傾きに合わせる')

    def test_anchor_matches_original_closed_mouth(self):
        story=json.loads((ROOT/'projects/short-001/storyboard.json').read_text(encoding='utf-8'))
        for pose,spec in story['mouths'].items():
            with self.subTest(pose=pose):
                image=Image.open(ROOT/story['poses'][pose]).convert('RGBA')
                image=image.crop(image.getchannel('A').point(lambda p:255 if p>24 else 0).getbbox())
                self.assertEqual(list(image.size),spec['landmarkSourceSize'])
                x0,y0,x1,y1=spec['closedMouthBox']
                self.assertAlmostEqual(spec['center'][0]*image.width,(x0+x1)/2,places=5)
                self.assertAlmostEqual(spec['center'][1]*image.height,(y0+y1)/2,places=5)
                region=np.array(image.crop((x0-2,y0-2,x1+2,y1+2)))[:,:,:3]
                ys,xs=np.where(region.max(axis=2)<220)
                self.assertGreater(len(xs),20)
                self.assertLess(abs((xs.min()+xs.max()+1)/2+x0-2-(x0+x1)/2),2)
                self.assertLess(abs((ys.min()+ys.max()+1)/2+y0-2-(y0+y1)/2),2)

    def test_only_mouth_changes_in_each_visible_pose(self):
        story=json.loads((ROOT/'projects/short-001/storyboard.json').read_text(encoding='utf-8'))
        for pose,spec in story['mouths'].items():
            with self.subTest(pose=pose):
                image=Image.open(ROOT/story['poses'][pose]).convert('RGBA')
                image=image.crop(image.getchannel('A').point(lambda p:255 if p>24 else 0).getbbox())
                closed=np.array(image)
                opened=np.array(mouth_only(image,1,spec))
                changed=np.any(opened!=closed,axis=2)
                self.assertTrue(changed.any())
                self.assertFalse(changed[:round(image.height*.165)].any(),'目・ハイライト領域は一切変更しない')
                x0,y0,x1,y1=[round(v*(image.width if i%2==0 else image.height)) for i,v in enumerate(spec['erase'])]
                changed[y0:y1,x0:x1]=False
                self.assertFalse(changed.any(),'口の矩形外は変更しない')
                np.testing.assert_array_equal(np.array(mouth_only(image,0,spec)),closed)

    def test_phonemes_follow_real_duration_and_pauses(self):
        q={'prePhonemeLength':.2,'postPhonemeLength':.3,'accent_phrases':[{'moras':[{'consonant':'h','consonant_length':.1,'vowel':'a','vowel_length':.3}],'pause_mora':{'vowel_length':.1}}]}
        cues=phoneme_cues(q,2,start=5)
        self.assertAlmostEqual(cues[-1]['end'],7)
        scene={'mouthCues':cues}
        for cue in cues:
            value=mouth_at(scene,(cue['start']+cue['end'])/2)
            if cue['phoneme']=='pau':self.assertEqual(value,0)
            else:self.assertGreater(value,0)
        self.assertEqual(mouth_at(scene,4),0)
        self.assertEqual(mouth_at(scene,7.1),0)


if __name__=='__main__':
    unittest.main()
