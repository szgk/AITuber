# 設定と決定事項はADRに残す

共通ルールは [AGENTS.md](../../AGENTS.md)、記録方法は [ADR 0002](../../docs/adr/0002-record-settings-and-decisions.md) に従う。

- 毎セッション開始時・再開時に `AGENTS.md`、`docs/adr/README.md`、[指摘の記録・引き継ぎルール](../../docs/adr/0017-always-record-user-feedback.md)、一覧の「ユーザー指摘の引き継ぎ」に載せた有効なADRの本文を必ず読む。その後、関連ADRを読む。
- ユーザーの指摘・修正要求・禁止事項・好みは大小を問わず必ずADRに残す。原文・適用範囲・対応・未解決点を同じ作業内で記録し、一覧の「ユーザー指摘の引き継ぎ」を更新する。修正だけで完了させず、会話や一時メモだけに残さない。
- キャラクター設定、世界観、動画制作方針、技術設計、運用の決定事項を `docs/adr/` に記録し、一覧を更新する。
- 未採用の提案と決定事項を区別する。
- 決定変更時は後継ADRを作り、元の判断経緯を残す。
- 画像・動画の制作・修正後は[空間整合性レビュースキル](../../.agents/skills/visual-continuity-review/SKILL.md)を必ず使い、実出力と拡大画像で接地・遮蔽などを検証して証拠を保存する。未レビュー・不合格のまま納品完了扱いにしない（[ADR 0018](../../docs/adr/0018-mandatory-visual-continuity-review.md)）。
