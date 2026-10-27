# 全身アニメーションの動画制作

配信用収録室とは独立した、台本から音声付きMP4を出力する制作経路。[ADR 0015](../../docs/adr/0015-full-body-short-production.md)に基づく。

初回作品は[隠れてご挨拶](../../projects/short-001/storyboard.json)。1080×1920・30fps、採用済みVOICEVOXの声、音声実尺の字幕、専用の立ち・しゃがみ・髪を直しながら顔を隠す仕草、会話表情、接地影、背景・机の遮蔽、カメラ移動を合成する。[ADR 0016](../../docs/adr/0016-hair-gesture-and-matched-background.md)に従い、お辞儀は使わず、背景も人物と同じ作画へ変更した。Live2Dや3Dの関節リグではなく、専用ポーズ画像と微小変形・ディゾルブによる2Dアニメーション。

## 実行

Python 3.12、Pillow、NumPy、FFmpegを使用する。Pythonは現在のCodex同梱ランタイムでも実行可能。VOICEVOXは既定 `http://127.0.0.1:50031`、`--engine` または `VOICEVOX_URL` で変更する。

```powershell
python tools/shorts/render.py --preview
python tools/shorts/render.py --reuse-audio
python tools/shorts/prepare_visual_review.py --output outputs/shorts/short-001-v4
python tools/shorts/verify.py --output outputs/shorts/short-001-v4
```

最初のコマンドは540×960の確認動画。次は音声を再生成せず1080×1920の完成動画を出力する。`--reuse-audio`は台詞・演出・声設定と音声ハッシュの一致を要求する。台本を変えた場合は指定せず生成する。エンジンバージョン・話者・声設定を含むキーで行単位のWAVをキャッシュする。

FFmpegは `FFMPEG` 環境変数、PATH、作業フォルダーの `.cache/shorts-python/imageio_ffmpeg/binaries/`、参考プロジェクト `D:/auto_movie/video-generator` の順に検索する。既存のRemotion同梱版はこの端末で起動時クラッシュが発生したため、初回出力には独立したバイナリを使用する。

```powershell
python -m pip install Pillow numpy
python -m pip install --target .cache/shorts-python imageio-ffmpeg==0.6.0
```

出力先は `outputs/shorts/short-001-v2/`。MP4、字幕SRT、音声WAV、各場面PNG、一覧JPEG、クレジット、再生HTML、入力・素材・出力のSHA-256を含むmanifestを保存する。音声を途中で切らず、台詞の実尺と指定した余韻で尺を計算する。最終音声は-16 LUFS・最大真ピーク-1.5 dBTPを目標にFFmpegで調整する。

## 次の作品を作る

初回ショートの最新出力は[ADR 0023](../../docs/adr/0023-mouth-axis-follows-face-tilt.md)によるv6（`outputs/shorts/short-001-v6/`）。`mouthAxis` と `angleDegrees` で原画の顔に対する口の向きを指定し、開口の輪郭・内側・舌をまとめて回転する。位置・音声・目元は維持する。検証には `verify.py --output outputs/shorts/short-001-v6` を使う。

口元の品質と位置は[ADR 0022](../../docs/adr/0022-smooth-mouth-and-pose-landmarks.md)によるv5（`outputs/shorts/short-001-v5/`）へ更新。口の限定領域を8倍で描画し、各原画の閉じ口の輪郭矩形を中心の基準にする。`test_lipsync.py` で姿勢ごとの位置、口以外の画素不変、音素と無音を確認する。

最新の初回ショートは[ADR 0021](../../docs/adr/0021-mouth-only-sync-foreground-and-reading.md)によるv4（`outputs/shorts/short-001-v4/`）。口だけを音素同期し、ライトの支柱を前景に含め、読み上げ用 `speechText` と字幕用 `text` を分ける。検証・証拠の再抽出には `--output outputs/shorts/short-001-v4` を指定する。`test_lipsync.py` で口以外の画素不変・可視ポーズ・音素と無音を検証できる。

現行の初回動画は[ADR 0019](../../docs/adr/0019-close-opening-and-bust-framing.md)によるv3。冒頭を机越しの寄り、その後をバストアップにし、`outputs/shorts/short-001-v3/`へ出力する。証拠の再抽出には `prepare_visual_review.py --output outputs/shorts/short-001-v3` を使う。

画像・動画の制作・修正後は、必ず[空間整合性レビュースキル](../../.agents/skills/visual-continuity-review/SKILL.md)を適用する。レンダラーは実MP4からフレームと足元拡大を自動抽出し、`visual-review.json` を未レビュー状態で作る。実際に画像を表示し、項目別所見・確認した証拠を記録してから検証する。未レビュー・不合格・別動画のレビュー・証拠の改変は `verify.py` を失敗させる。技術検査の合格と視覚合格を分け、未完了のものを完成扱いにしない。

`projects/`へ台本JSON、`assets/shorts/`へ画像を追加し、`--project`で選択する。各場面に専用ポーズ、足元の座標、全身の高さ、カメラ倍率と注視点、字幕、台詞、余韻を指定する。画像は必要な動き・画角ごとに参照原画から生成し、単一の立ち絵を縮めてしゃがみの代用にしない。人物は実アルファ付きPNGを使う。

背景は人物を置ける床を含めて生成する。前景の机などは背景の同一ピクセルを `foregroundPolygon` で再合成し、キャラクターを遮蔽する。カメラは人物と背景の合成後に動かす。全身の比率と足元を揃え、位置と影で同じ空間へ接地させる。背景を変更したら前景ポリゴンも調整する。

生成そのものは対話中のimage_genで行う。レンダラーが画像APIを無人で呼ぶ機能は含まない。声・衣装・人物設定は既存ADRに従い、各作品の台詞や部屋は恒久設定と区別する。

書き出しは明示コマンドでのみ行い、公開・投稿は行わない。完成物はローカルの `index.html` またはMP4で確認できる。
