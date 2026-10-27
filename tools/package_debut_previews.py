"""目視済みの無音確認動画15本を記録・検証し、一覧とZIPにまとめる。"""
from pathlib import Path
import hashlib, html, json, re, subprocess, sys, zipfile,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent/'shorts'))
from render import find_binary

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
root=args.output.resolve() if args.output else Path(__file__).resolve().parents[1]/'outputs/debut-motion-preview'
titles=['頭だけ、こんにちは','昨日より、少し','手だけ先に','かわいい物には近づける','好きなものは、はっきり','笑ったところだけ','好きな話だと長くなる','片目で、こんにちは','顔を見て話せた日','犬の話なら','ちゃんと見せたいので','何も言わない時間も','あいさつを、一つ選ぶ','明日は、ご挨拶','ちゃんと、はじめまして']
specific=['頭頂部が小さく上下し、右上のくせ毛が揺れる。顔は見せない。','少し高くのぞく頭頂部と右上のくせ毛の揺れ。','顔を隠したまま挨拶の片手を振る。','花柄カードを持つ手と髪を小さく動かす。','サーモンのカードを見せる手と髪の動き。','目を出さない口元の寄りで開閉。','目を出さない口元の寄りで開閉。旧口が残ったピンクの矩形は修正後の実動画で消失を確認。','目隠しから少し顔を寄せて覗き込み、ハイライトが小さく揺れる。','両目の渦巻きが回転。閉口は緊張した波形、口角軸は右上がり。','目を細めた笑顔と口の開閉、元の口の中心を再計測して調整。','襟元の手の小さな動き、通常目のハイライトの揺れ。','目を閉じた穏やかな表情と、顔の傾きに沿う口の開閉。','頬へ指を添えた姿勢を保ち、目のハイライトが揺れる。','渦巻きが回転し、涙は小さな白い通常の楕円。両手は胸元で小さく動く。','顔の横で片手を振る。5本の指を維持し、口は顔の傾きへ追従。']
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
smooth_revision=root.name=='debut-motion-preview-v3'
rigid_revision=root.name in ('debut-motion-preview-v2','debut-motion-preview-v3')
if rigid_revision:
 specific[2]='手の輪郭・指間距離を保ち、手首を支点にパーツ全体を回転して挨拶する。'
 specific[3]='花柄カードを持つ手の形を固定し、手への局所変形を除去。'
 specific[4]='サーモンのカードを持つ手の形を固定し、手への局所変形を除去。'
 specific[10]='襟元の手の形を固定。目のハイライトと顔・髪の動きは維持。'
 specific[12]='頬に添えた指と手の形を固定。目のハイライトの揺れを維持。'
 specific[13]='胸元の両手は形を固定。ぐるぐる目と小さな白い楕円の涙を維持。'
 specific[14]='手全体を手首から剛体回転し、指・掌を歪ませずに挨拶する。'
cards=[];catalog=[]
videos=sorted(root.glob('*/video.mp4'))
assert len(videos)==15
for i,video in enumerate(videos):
 folder=video.parent
 evidence=[folder/'review/full-sequence.png',folder/'review/detail-04.png']
 if rigid_revision and not smooth_revision and i+1 in (3,15):evidence.append(folder/'review/hand-sequence.png')
 if smooth_revision and i+1>=6:evidence.append(folder/'review/mouth-sequence.png')
 assert all(p.exists() for p in evidence)
 observations=[('接地・床','対象外','全編に足・床が描かれていない。'),('遮蔽・前後関係','合格','実動画の前景・髪・手・顔の境界を確認。家具は存在せず対象外。'),('身体・姿勢','合格','原画の身体を局所的に変形。手や顔の増殖、切断なし。'+specific[i]),('時間・カメラ','合格','全6秒の0.1秒刻み60連続フレームで、開始・中間・終了と動作の往復を確認。飛びや二重像なし。'),('作画・光','合格','太い線とセル塗りを維持。口の方向が顔に沿い、旧口や大きな色の矩形が残らない。'),('指摘・画面','合格',specific[i]+' 顔中心の既定構図を維持。')]
 review={'overall':'合格','scope':'無音・テスト用口開閉の動作確認版としての視覚判定。声付き本編・音声同期の判定ではない。','target':'video.mp4','sha256':digest(video),'inspectedEvidence':[{'path':str(p.relative_to(folder)),'sha256':digest(p)} for p in evidence],'checks':[{'item':a,'status':b,'findings':c,'evidence':[str(p.relative_to(folder)) for p in evidence]} for a,b,c in observations],'unresolved':[]}
 if rigid_revision:
  if smooth_revision and i+1<=5:
   old=root.parent/'debut-motion-preview-v2'/folder.name/'visual-review.json'
   previous=json.loads(old.read_text(encoding='utf-8'))
   assert previous['sha256']==digest(video)
   review=previous;review['reviewReuse']='動画SHA-256が手パーツ修正版と同一。前回の実出力レビューを引き継ぐ。'
  elif i+1 in (3,15):
   review['checks'][1]['findings']='旧手の残像がなく、袖口と手首が接続。指・掌の形を保ったまま手が回転する。'
   review['checks'][2]['findings']='指5本、指間距離と掌の形を保持。メッシュによる伸縮なし。03の小指も画面内に収まる。'
  elif i+1 in (4,5,11,13,14):review['checks'][2]['findings']='持つ・添える・握る手の形を固定し、局所的な伸縮を除去。'
  elif not smooth_revision:
   old=root.parent/'debut-motion-preview'/folder.name/'visual-review.json'
   previous=json.loads(old.read_text(encoding='utf-8'))
   assert previous['sha256']==digest(video)
   review=previous;review['reviewReuse']='動画SHA-256が旧版と同一。前回の実出力のレビューを引き継ぐ。'
 if smooth_revision and i+1>=6:
  review['checks'][3]['findings']='実MP4の全編60連続フレームと、口元の30fps・22連続フレームを確認。閉じ口から開口まで中間形状が入り、線と楕円の急な切替がない。'
  review['checks'][5]['findings']+=' 開口量と輪郭を連続補間。顔の傾き・口の位置・目元と手の動きを維持。'
 (folder/'visual-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
 result=subprocess.run([find_binary('ffmpeg'),'-hide_banner','-i',str(video),'-vf','showinfo','-an','-f','null','-'],capture_output=True)
 log=result.stderr.decode(errors='replace')
 frames=len(re.findall(r'Parsed_showinfo.*\bn:\s*\d+\s+pts:',log))
 assert result.returncode==0 and frames==180,(folder.name,frames)
 assert '1080x1080' in log and '30 fps' in log
 check={'decode':'合格','frames':frames,'size':[1080,1080],'fps':30,'duration':6,'audio':'無音の確認版','sha256':digest(video)}
 (folder/'technical-check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
 meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));meta['status']='無音確認版として視覚・技術レビュー済み';meta['audio']='無音。口の開閉は演技確認用。';(folder/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 rel=video.relative_to(root).as_posix();day=f'D−{14-i}' if i<14 else '当日'
 title=f'{i+1:02} ／ {day} ／ {titles[i]}'
 cards.append(f'<article><h2>{html.escape(title)}</h2><video controls preload="metadata" src="{rel}" poster="{folder.name}/review/frame-011.png"></video><p>{specific[i]}</p><a href="{rel}" download="{i+1:02}-d-{14-i:02}.mp4">MP4を保存</a></article>')
 catalog.append({'number':i+1,'day':day,'title':titles[i],'video':rel,**check})
page='<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>望月京子・15本の無音動作確認動画</title><style>body{background:#eef0f2;color:#233449;font-family:system-ui,sans-serif;margin:32px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:24px}article{background:white;padding:18px;border-radius:14px}video{width:100%;border-radius:8px}h2{font-size:17px}p{line-height:1.7}a{color:#235baa}</style><h1>望月京子・15本の動作確認動画</h1><p><strong>全15本は無音の確認版です。</strong>各6秒・1080×1080・30fps。口の開閉はテスト用で、声付き本編と音声同期は未完成です。音声生成コマンドは応答を得られていません。</p><p><a href="debut-motion-15-videos.zip" download>15本のMP4をまとめて保存</a></p><main>'+''.join(cards)+'</main></html>'
if rigid_revision:page=page.replace('<h1>','<h1>口の開閉を滑らかにした版：' if smooth_revision else '<h1>手パーツ修正版：',1)
(root/'index.html').write_text(page,encoding='utf-8')
(root/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
readme='# 望月京子：15本の無音動作確認動画\n\n各6秒・1080×1080・30fps。番号01〜15はデビュー14日前〜当日の各ラフに対応します。\n\n口は動作確認用の開閉です。採用済みVOICEVOXでの声付き本編は、ローカル接続コマンドが応答せず未完成です。別の声や既存の別台詞の音声では代用していません。\n\n各動画の全編60連続フレームと顔・手・髪の詳細を実MP4から確認し、視覚レビュー・技術検証・ハッシュを各フォルダーに保存しました。\n\n一覧のindex.htmlを開くと個別再生・保存できます。\n'
(root/'README.md').write_text(readme,encoding='utf-8')
with zipfile.ZipFile(root/'debut-motion-15-videos.zip','w',zipfile.ZIP_DEFLATED) as z:
 z.write(root/'README.md','README.md');z.write(root/'catalog.json','catalog.json')
 for video in videos:z.write(video,f'{video.parent.name}.mp4')
print('15本の無音動画を検証・記録・梱包しました。',flush=True)
