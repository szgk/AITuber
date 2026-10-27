"""対象や証拠が変わった場合に視覚レビューを再要求することを検証する。"""
import json
from pathlib import Path
import tempfile
import unittest
from render import digest
from visual_review_gate import REQUIRED, validate


class VisualGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.movie = self.folder/'movie.mp4'
        self.movie.write_bytes(b'video-version-one')
        self.evidence = self.folder/'frame.png'
        self.evidence.write_bytes(b'frame-version-one')
        self.report = {'status':'合格','outputHash':digest(self.movie),'evidence':{'frame.png':digest(self.evidence)},'inspectedEvidence':['frame.png'],'findings':[],'checks':[{'name':name,'status':'合格','notes':'対応する実出力を確認した所見','evidence':['frame.png']} for name in REQUIRED]}

    def save(self):
        (self.folder/'visual-review.json').write_text(json.dumps(self.report,ensure_ascii=False),encoding='utf-8')

    def test_review_required(self):
        self.assertTrue(validate(self.folder,self.movie))

    def test_complete_report(self):
        self.save()
        self.assertEqual(validate(self.folder,self.movie),[])

    def test_changed_video_rejected(self):
        self.save()
        self.movie.write_bytes(b'video-version-two')
        self.assertTrue(validate(self.folder,self.movie))

    def test_changed_evidence_rejected(self):
        self.save()
        self.evidence.write_bytes(b'frame-version-two')
        self.assertTrue(validate(self.folder,self.movie))

    def test_uninspected_evidence_rejected(self):
        self.report['inspectedEvidence'] = []
        self.save()
        self.assertTrue(validate(self.folder,self.movie))

    def test_unresolved_issue_rejected(self):
        self.report['findings'] = [{'notes':'足が机の前に出て見える'}]
        self.save()
        self.assertTrue(validate(self.folder,self.movie))

    def test_pending_and_failed_rejected(self):
        for status in ['未レビュー','不合格']:
            self.report['status'] = status
            self.save()
            self.assertTrue(validate(self.folder,self.movie))


if __name__ == '__main__':
    unittest.main()
