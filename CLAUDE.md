# Claude プロジェクト設定

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