#!/bin/bash
# Ollama統合セットアップスクリプト
# LoRAモデルのGGUF変換からOllama登録まで自動化

set -e

echo "🚀 AITuber望月京子 Ollama統合セットアップ"
echo "============================================"

# 設定
LORA_PATH="./outputs/lora_model"
OLLAMA_DIR="./ollama_models"
MODEL_NAME="aituber-kyoko"

# 前提条件チェック
echo "📋 前提条件をチェック中..."

# LoRAモデルの存在確認
if [ ! -d "$LORA_PATH" ]; then
    echo "❌ LoRAモデルが見つかりません: $LORA_PATH"
    echo "   先にLoRA学習を実行してください: npm run train"
    exit 1
fi

# Ollamaのインストール確認
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollamaがインストールされていません"
    echo "   インストール方法: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Ollamaサーバーの起動確認
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  Ollamaサーバーが起動していません"
    echo "   以下のコマンドで起動してください: ollama serve"
    echo "   起動後、このスクリプトを再実行してください"
    exit 1
fi

echo "✅ 前提条件チェック完了"

# ステップ1: GGUF変換の準備
echo ""
echo "📦 ステップ1: GGUF変換の準備"
echo "--------------------------------"

if [ ! -f "convert_to_gguf.py" ]; then
    echo "❌ convert_to_gguf.py が見つかりません"
    exit 1
fi

echo "🔄 LoRAモデルをGGUF形式に変換中..."
python convert_to_gguf.py --lora-path "$LORA_PATH" --output-dir "$OLLAMA_DIR"

# ステップ2: GGUF変換の実行
echo ""
echo "🔧 ステップ2: GGUF変換の実行"
echo "--------------------------------"

CONVERSION_SCRIPT="$OLLAMA_DIR/convert_to_gguf.sh"
if [ -f "$CONVERSION_SCRIPT" ]; then
    echo "🚀 GGUF変換スクリプトを実行中..."
    chmod +x "$CONVERSION_SCRIPT"
    bash "$CONVERSION_SCRIPT"
else
    echo "❌ 変換スクリプトが生成されませんでした"
    exit 1
fi

# ステップ3: Ollamaへのモデル登録
echo ""
echo "📝 ステップ3: Ollamaへのモデル登録"
echo "--------------------------------"

MODELFILE="$OLLAMA_DIR/Modelfile"
if [ ! -f "$MODELFILE" ]; then
    echo "❌ Modelfileが見つかりません: $MODELFILE"
    exit 1
fi

# 既存のモデルを削除（存在する場合）
if ollama list | grep -q "$MODEL_NAME"; then
    echo "🗑️  既存のモデルを削除中: $MODEL_NAME"
    ollama rm "$MODEL_NAME"
fi

echo "📝 Ollamaにモデルを登録中: $MODEL_NAME"
cd "$OLLAMA_DIR"
ollama create "$MODEL_NAME" -f Modelfile
cd - > /dev/null

# ステップ4: モデルのテスト
echo ""
echo "🧪 ステップ4: モデルのテスト"
echo "--------------------------------"

echo "💬 簡単なテストを実行中..."
ollama run "$MODEL_NAME" "自己紹介をしてください" | head -n 5

# ステップ5: APIテストの実行
echo ""
echo "🔍 ステップ5: APIテストの実行"
echo "--------------------------------"

if [ -f "ollama_test.py" ]; then
    echo "🧪 包括的なAPIテストを実行中..."
    python ollama_test.py --model "$MODEL_NAME"
else
    echo "⚠️  ollama_test.py が見つかりません。基本テストのみ実行しました。"
fi

# 完了メッセージ
echo ""
echo "🎉 セットアップ完了！"
echo "===================="
echo ""
echo "✅ AITuber望月京子モデルがOllamaに登録されました"
echo ""
echo "📚 使用方法:"
echo "  1. チャット: ollama run $MODEL_NAME"
echo "  2. API: curl http://localhost:11434/api/generate -d '{\"model\": \"$MODEL_NAME\", \"prompt\": \"こんにちは\"}'"
echo "  3. テスト: python ollama_test.py --chat"
echo ""
echo "🔧 管理コマンド:"
echo "  - モデル一覧: ollama list"
echo "  - モデル削除: ollama rm $MODEL_NAME"
echo "  - ログ確認: ollama logs"
echo ""
echo "💡 ヒント: 詳細なテストを実行するには python ollama_test.py を実行してください"