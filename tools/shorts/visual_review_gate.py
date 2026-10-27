"""実動画と結び付いた視覚レビューが完了しているかを検査する。"""
import json
from pathlib import Path
from render import digest

REQUIRED = {'接地・床','遮蔽・前後関係','身体・姿勢','時間・カメラ','作画・光','指摘・画面'}


def validate(folder, movie):
    file = Path(folder)/'visual-review.json'
    if not file.exists():
        return ['視覚レビュー記録がありません']
    report = json.loads(file.read_text(encoding='utf-8'))
    errors = []
    if report.get('outputHash') != digest(Path(movie)):
        errors.append('視覚レビューの対象動画が異なります')
    if report.get('status') != '合格':
        errors.append(f'視覚レビュー判定: {report.get("status","未記録")}')
    if report.get('findings'):
        errors.append('未解消の視覚所見があります')
    checks = report.get('checks',[])
    if {c.get('name') for c in checks} != REQUIRED or len(checks) != len(REQUIRED):
        errors.append('必須レビュー項目に不足または重複があります')
    inspected = set(report.get('inspectedEvidence',[]))
    for check in checks:
        if check.get('status') != '合格' or not check.get('notes') or not check.get('evidence'):
            errors.append(f'未合格または証拠不足: {check.get("name")}')
        for reference in check.get('evidence',[]):
            evidence = (Path(folder)/reference).resolve()
            if not evidence.is_relative_to(Path(folder).resolve()):
                errors.append('証拠パスが出力フォルダー外です')
            elif reference not in inspected or reference not in report.get('evidence',{}) or not evidence.is_file() or digest(evidence) != report['evidence'][reference]:
                errors.append(f'未確認・欠落・改変された証拠: {reference}')
    return errors
