"""完成したMP4のデコード・解像度・音声・制作入力を検証する。"""
from pathlib import Path
import argparse
import json
import re
import subprocess
from render import ROOT, digest, find_binary
from visual_review_gate import validate
from check_lipsync import check as check_speech


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/shorts/short-001-v2')
    args = parser.parse_args()
    folder = ROOT / args.output
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    movie = ROOT/manifest['output']
    assert manifest['status'] == '書き出し完了'
    assert manifest['outputHash'] == digest(movie), '動画がmanifestと一致しません'
    renderer_matches = manifest['rendererHash'] == digest(Path(__file__).with_name('render.py'))
    for file,sha in manifest.get('rendererDependencies',{}).items():
        assert digest(ROOT/file) == sha, f'描画依存ファイルが変更されています: {file}'
    assert manifest['storyboardHash'] == digest(ROOT/manifest['project'])
    for path, sha in manifest['assets'].items():
        assert digest(ROOT/path) == sha, f'素材改変: {path}'
    scenes = manifest['timeline']
    for index, scene in enumerate(scenes):
        assert scene['start'] < scene['speechEnd'] < scene['end'] <= manifest['duration']+.001
        if index:
            assert abs(scenes[index-1]['end']-scene['start']) < .001
        keys = scene.get('poseSequence',[])
        if keys:
            assert keys[0]['at'] == 0
            assert all(0 <= key['at'] < 1 for key in keys)
            assert all(a['at'] < b['at'] for a,b in zip(keys,keys[1:]))
            assert all(key['pose'] in json.loads((ROOT/manifest['project']).read_text(encoding='utf-8'))['poses'] for key in keys)
    assert all(scene['pose'] != 'bow' and all(key['pose'] != 'bow' for key in scene.get('poseSequence',[])) for scene in scenes)
    result = subprocess.run([find_binary('ffmpeg'),'-hide_banner','-i',str(movie),'-af','loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json','-f','null','-'],capture_output=True)
    log = result.stderr.decode('utf-8',errors='replace')
    assert result.returncode == 0, log
    assert f'{manifest["width"]}x{manifest["height"]}' in log
    assert 'h264' in log and 'aac' in log
    match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)',log)
    assert match, log
    duration = int(match[1])*3600+int(match[2])*60+float(match[3])
    assert abs(duration-manifest['duration']) < .15
    measured = json.loads(log[log.rfind('{'):log.rfind('}')+1])
    assert -18 < float(measured['input_i']) < -14, measured
    assert float(measured['input_tp']) < -.8, measured
    report = {'status':'合格','output':str(movie.relative_to(ROOT)),'duration':duration,'dimensions':[manifest['width'],manifest['height']],'fps':manifest['fps'],'codecs':['H.264','AAC'],'integratedLufs':float(measured['input_i']),'truePeakDbtp':float(measured['input_tp']),'checks':['映像と音声を全尺デコード','解像度とコーデック','音声実尺と動画尺の一致','字幕タイムラインの連続性','音声ラウドネスとピーク','素材・台本・レンダラー・動画のハッシュ一致'],'visualReview':'場面一覧と抽出画像で頭頂部の遮蔽・全身の収まり・足元・字幕の収まりを確認。音声の人間による試聴は未実施。'}
    visual_errors = validate(folder,movie)
    if all(s.get('mouthCues') for s in scenes):
        report['speechChecks'] = check_speech(folder)
    report['checks'][-1] = '素材・台本・動画のハッシュ一致（現行レンダラーとの差異は別記）'
    report['status'] = '不合格' if visual_errors else '合格'
    report['technicalStatus'] = '合格'
    report['rendererMatchesCurrentSource'] = renderer_matches
    report['visualReview'] = {'errors':visual_errors,'report':'visual-review.json'}
    (folder/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if visual_errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
