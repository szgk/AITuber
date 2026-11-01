# Architecture Decision Records

このディレクトリには、AITuber のキャラクター設定、世界観、動画制作方針、技術設計、運用の決定事項とその影響を日本語で記録する。

- [0001: 望月京子 AITuber の現行構成と仕様](0001-aituber-current-architecture.md)
- [0002: 設定と決定事項をADRで管理する](0002-record-settings-and-decisions.md)
- [0003: デザインver1のブラウザアバターと動画収録](0003-browser-avatar-recording-studio.md)
- [0003: 動画企画を定期的に自動作成する](0003-recurring-video-ideation.md)
- [0004: 合成音声を用いた動画制作を先行する](0004-video-first-production-scope.md)
- [0005: 気弱さと小さな震えを持つ声の方向性](0005-timid-voice-direction-and-samples.md)
- [0006: 創作キャラクターへの声当てを許可するVOICEVOX音声の比較](0006-voicevox-original-character-voice-samples.md)
- [0007: 京子の声に比較サンプル⑥を採用する](0007-adopt-kyoko-voice.md)
- [0008: デビューまでの準備と告知の流れ](0008-debut-preparation-flow.md)
- [0009: ロゴをポップな方向で試作する](0009-logo-pop-direction.md)
- [0010: ロゴはポップな書体と紺色ベースにする](0010-logo-pop-type-navy-color.md)
- [0011: 「月」の花びらを白枠と紺枠で縁取る](0011-logo-petal-outline.md)
- [0012: 第6案を望月京子のロゴに採用する](0012-adopt-kyoko-logo.md)
- [0013: デビュー前の投稿で姿と性格を段階的に見せる](0013-gradual-debut-reveal.md)
- [0014: 動画の背景・字幕・演出を収録室に用意する](0014-video-visual-kit.md)
- [0015: 専用ポーズと空間レイヤーによる全身ショート制作](0015-full-body-short-production.md)
- [0016: お辞儀を使わず顔を隠す仕草と背景の作画を合わせる](0016-hair-gesture-and-matched-background.md)
- [0017: ユーザーの指摘を必ず記録し毎セッション引き継ぐ](0017-always-record-user-feedback.md)
- [0018: 空間・身体・動きの矛盾を必ず視覚レビューする](0018-mandatory-visual-continuity-review.md)
- [0019: 冒頭を机越しの寄り、その後をバストアップにする](0019-close-opening-and-bust-framing.md)
- [0021: 口だけを音素同期しライトの前後関係と読みを修正する](0021-mouth-only-sync-foreground-and-reading.md)
- [0022: 口の輪郭を滑らかにし元画像の口を基準に位置を合わせる](0022-smooth-mouth-and-pose-landmarks.md)
- [0023: 開いた口の向きを原画の顔の傾きへ合わせる](0023-mouth-axis-follows-face-tilt.md)
- [0024: 手は歪ませずパーツ単位で揺らす](0024-rigid-hand-part-animation.md)
- [0025: 口の開閉を時間と輪郭の両方で滑らかにつなぐ](0025-smooth-mouth-opening.md)

- [0026: PCの中で目覚めた元OLが元に戻る方法を探す](0026-kyoko-awakening-in-pc.md)

## ユーザー指摘の引き継ぎ

- [0029: 現行デザインと設定を維持してVTuber風の描画にする](0029-vtuber-illustration-style.md) — 髪型・服装・設定を維持。薄い頬染め、ぐるぐる目、大きな汗と顎の滴。青は薄く、おでこから目の中央付近まで。修正版v6、具体案全体は確認待ち。

- [0028：スキル共通化と.env](0028-shared-skills-file-env.md) — スキル使用時に共通・個別.envを毎回読み、Windowsユーザー環境変数を設定原本にしない。

- [0027: 動画案の自動生成を一時停止する](0027-pause-video-ideation.md) — 自動化 `automation-3` は停止済み。ユーザーの再開指示まで自動生成しない。

- [0026: PCの中で目覚めた元OLが元に戻る方法を探す](0026-kyoko-awakening-in-pc.md) — 現在の行動範囲は最初の部屋・PC画面・ネット。段階的な機能実装は本人の知らない制作側の方針。

- [0025: 口の開閉を滑らかにつなぐ](0025-smooth-mouth-opening.md) — 音素間の開口量と閉じ口から開口への輪郭を連続補間し、顔の傾きと無音時の閉口を維持する。

- [0024: 手は歪ませずパーツ単位で揺らす](0024-rigid-hand-part-animation.md) — 指・掌の形を保ち、手首を支点とする回転・並進で動かす。手の局所メッシュ変形は廃止する。

- [0023: 口の向きを顔の傾きに合わせる](0023-mouth-axis-follows-face-tilt.md) — 初回ショートv6。各原画の口角軸で開口全体を回転し、顔と同じ変形へ渡す。位置・目元・音声は維持する。

- [0022: 告知ラフを採用し、顔・目・手・跳ね毛が動く声付き動画へ進む](0022-debut-video-character-animation.md) — 涙は小さな楕円。ぐるぐる目・うるうる・手振り・覗き込み・右上のくせ毛を動かす。口の軸は原画の顔の傾きと頭の動きへ追従させ、水平に固定しない。

- [0022: 口のギザつき・姿勢ごとの位置ずれを確認して直す](0022-smooth-mouth-and-pose-landmarks.md) — 口を高解像度描画し、各原画の閉じ口を位置基準にする。顔の他の部分は変更しない。

- [0021: 口パクで目を変えない・音声同期・ライト支柱を髪より前・「おはなし」の読み](0021-mouth-only-sync-foreground-and-reading.md) — 初回ショートv4に反映。手や机で隠れた口を上から描かない。

- [0020: 顔アップ・豊かな表情の画像ラフを先行する](0020-face-closeup-expression-roughs.md) — 顔中心、画像ラフを先行。09はぐるぐる目、14はぐるぐる目と涙。頬染めは薄くし、涙は汗と同じ白ベタ・黒輪郭の漫画表現にする。

- [0019: デビュー14日前からの投稿原稿と静止画を保存する](0019-debut-post-assets.md) — 文書と画像の制作・保存依頼。日割りは提案、動画・公開日時は未確定。

毎セッション開始時・再開時に、以下の有効なADRの本文を必ず読む。新しい指摘を記録したらこの一覧にも追加し、置換時は後継を示す。

- [0019: 冒頭は添付画像程度に寄り、その後はバストアップ](0019-close-opening-and-bust-framing.md) — 初回ショートの現行画角。奥行き・前景マスクと二重像も修正し、v3を別出力する。
- [0018: 足・床・家具などの矛盾をスキルで必ずレビューする](0018-mandatory-visual-continuity-review.md) — 画像・動画の制作・修正後、実出力と拡大画像の視覚レビューを必須化。旧v2は不合格として保存し、v3の修正はADR 0019を参照。
- [0017: 指摘は大小を問わず必ずADRに記録し、毎セッション読み直す](0017-always-record-user-feedback.md) — 全作業に適用する記録・引き継ぎルール。
- [0016: お辞儀を使わず、顔を隠す仕草にし、背景も人物と同じ作画にする](0016-hair-gesture-and-matched-background.md) — 初回ショートへの指摘。髪を直すという解釈は暫定であり、ユーザーによる語義の確定とは区別する。

記録方法と状態の扱いはADR 0002に従う。提案を採用済みと混同しない。

既存実装を後から記録する ADR は「現行実装の記録」と明記する。新しい設計判断は別番号で追加し、変更する ADR への参照を残す。未実装の構想は、採用済みの機能と区別する。

- [0024: Twitterアイコンはぐるぐる目、ヘッダーはロゴを使う](0024-twitter-profile-assets.md) — 有効なユーザー指定。具体的な画像案は確認待ち。
