# ADR 0028：スキル原本を共通化し.envを毎回参照する

- 状態：採用・移動済み
- 日付：2026-10-29
- 決定者：ユーザー

## 背景と指示

ユーザー指定「全スキル移動」、続いて「ユーザー環境変数ではなくAIエージェントが参照する環境変数をenvファイルとしてスキルのフォルダ内に定義してそれを毎回参照するようにして」。

## 決定と対応

`visual-continuity-review`の正本をホーム配下の`.agents/skills/visual-continuity-review`へ統一し、このリポジトリの`.agents/skills/visual-continuity-review`はその正本へのジャンクションにした。移動前の全ファイルは共通原本と一致し、バックアップを保存した。

共通`skills/.env`と個別`visual-continuity-review/.env`を使用ごとに読み、相対パスをファイル設定から解決する。Windowsのユーザー環境変数を設定原本にしない。全39スキルで共通原本と.envの参照を確認した。

## 維持する仕様と影響

ADR 0018の納品前視覚レビュー義務、判定基準、証拠の保存は維持。旧パスからも同じファイルを読めるため既存の開始時参照は有効。動画・人物設定・素材・自動化の停止状態は変更していない。

詳細な移動・検証記録は`D:/original/inusereas/.claude/adr/ADR-0016-all-skills-migration-and-file-env.md`。
