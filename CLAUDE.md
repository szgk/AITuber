# Claude プロジェクト設定

## 設定と決定事項の記録

共通作業ルールは [AGENTS.md](AGENTS.md) に従う。

人物・背景の合成画像や動画の制作・修正後、納品前に[空間整合性レビュースキル](.agents/skills/visual-continuity-review/SKILL.md)を必ず読み、実出力と足元・家具境界の拡大画像を確認する。証拠と所見を保存し、未レビュー・不合格のまま完成扱いにしない。

- 毎セッション開始時・再開時にAGENTS.md、ADR一覧、[指摘の記録・引き継ぎルール](docs/adr/0017-always-record-user-feedback.md)、一覧の「ユーザー指摘の引き継ぎ」に載せた有効なADRの本文を必ず読む。
- ユーザーの指摘・修正要求・禁止事項・好みは、大小を問わず必ず同じ作業内でADRに残す。修正だけで完了させず、原文・適用範囲・対応・未解決点を記録する。会話や一時メモだけで済ませない。
- キャラクター設定、世界観、動画制作方針、技術設計、運用の決定事項は `docs/adr/` に連番のADRとして記録し、[ADR一覧](docs/adr/README.md) を更新する。
- 作業前に関連ADRを確認する。会話や企画メモだけに決定を残さない。
- 提案と採用済みを区別し、未採用の案を確定設定にしない。
- 決定変更は後継ADRで記録し、元のADRとの参照と履歴を残す。
- 記録方法は [ADR 0002](docs/adr/0002-record-settings-and-decisions.md) を参照する。

## 言語設定
このプロジェクトでは日本語でのコミュニケーションを行います。以下の点に注意してください：

- すべての会話は日本語で行う
- コメントやドキュメントも日本語で記述
- エラーメッセージや説明も日本語で提供
- 技術用語は必要に応じて英語のまま使用可能

## Git操作ガイドライン

### 基本設定
- リポジトリURL: `git@github.com:szgk/AITuber.git`
- メインブランチ: `main`
- 大文字小文字を区別しない設定（core.ignorecase = true）

### コミットメッセージ
コミットメッセージは以下の形式で記述：

```
<種別>: <概要>

<詳細説明（必要に応じて）>

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
```

種別の例：
- `feat`: 新機能追加
- `fix`: バグ修正
- `docs`: ドキュメント更新
- `style`: コードスタイル修正
- `refactor`: リファクタリング
- `test`: テスト追加・修正
- `chore`: その他の変更

### Gitコマンド実行時の注意事項
1. コミット前に必ず以下を確認：
   - `git status` で変更ファイルを確認
   - `git diff` で変更内容を確認
   - `git log --oneline -5` で最近のコミット履歴を確認

2. ステージングとコミット：
   ```bash
   git add <ファイル名>
   git commit -m "$(cat <<'EOF'
   feat: 新機能の説明
   
   🤖 Generated with Claude Code
   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
   ```

3. プルリクエスト作成：
   ```bash
   gh pr create --title "PR タイトル" --body "$(cat <<'EOF'
   ## 概要
   - 変更内容の箇条書き
   
   ## テスト方法
   - テスト手順の説明
   
   🤖 Generated with Claude Code
   EOF
   )"
   ```

## 作業時の推奨事項
1. 変更を加える前に、関連ファイルの現在の状態を確認
2. コミットは小さく、頻繁に行う
3. 機能ごとにブランチを作成することを検討
4. テスト可能な場合は、変更後にテストを実行
