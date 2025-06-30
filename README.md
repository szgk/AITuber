# AITuber

LoRA（Low-Rank Adaptation）を使用したAITuberシステム

## 概要

このプロジェクトは、Qwen2.5ベースモデルにLoRAファインチューニングを適用して、AITuberキャラクターを作成するシステムです。

## プロジェクト構成

```
AITuber/
├── data/
│   ├── character_profile.json      # キャラクター設定
│   └── training/                   # 学習データ
│       ├── train.jsonl            # 訓練用データ（127件）
│       ├── validation.jsonl       # 検証用データ（15件）
│       └── prepare_dataset.py     # データ準備スクリプト
├── config.yaml                    # 学習設定ファイル
├── train_lora.py                  # LoRA学習スクリプト
├── inference.py                   # 推論スクリプト
├── chat_interface.py              # チャットインターフェース
├── test_model.py                  # モデルテストスクリプト
├── run.py                         # 統合実行スクリプト
└── requirements.txt               # 必要パッケージ
```

## セットアップ

### 1. 環境準備

```bash
# リポジトリをクローン
git clone <repository-url>
cd AITuber

# 仮想環境の作成
python3 -m venv venv

# 仮想環境の有効化
# Windows:
venv\Scripts\activate
# Linux/Mac/WSL:
source venv/bin/activate

# パッケージのインストール
pip install -r requirements.txt
```

#### ⚠️ トラブルシューティング: Python環境

**pyenvエラーの対処法**:
```bash
# エラー例: /mnt/c/Users/.../pyenv/shims/python: cannot execute: required file not found

# 1. システムPythonを使用（推奨）
/usr/bin/python3 -m venv venv

# 2. または利用可能なPythonを確認
which python3
python3 --version

# 3. pyenv設定をリセット（必要に応じて）
unset PYENV_ROOT
export PATH="/usr/bin:$PATH"
```

**WSL環境での注意点**:
- Windows側のpyenvではなく、WSL内のPythonを使用してください
- `python3`コマンドを明示的に使用することを推奨します

### 2. 環境確認

```bash
# 環境テスト
python test_lora_setup.py

# または
python run.py check-env
```

## 使用方法

### npmスクリプト（推奨・venv自動使用）

**npmスクリプトは自動的にvenv環境を使用するため、仮想環境の手動有効化は不要です。**

#### 基本コマンド

```bash
# 利用可能なコマンド一覧を表示
npm run help

# 環境チェック（GPU、CUDA、パッケージの確認）
npm run check-env

# データセット準備（必要に応じて）
npm run prepare-data
```

#### 学習関連コマンド

```bash
# 通常のLoRA学習（3エポック）
npm run train

# 高速学習（1エポック、テスト用）
npm run train:quick
```

#### チャットとインターフェース

```bash
# 対話式チャットインターフェース
npm run chat

# デモモード（プリセット質問）
npm run chat:demo

# 推論モード（バッチ処理）
npm run inference

# 推論デモ
npm run inference:demo
```

#### モデルテスト

```bash
# 包括的なモデルテスト
npm run test

# 高速テスト（基本項目のみ）
npm run test:quick

# 基本応答テストのみ
npm run test:basic

# 人格一貫性テストのみ
npm run test:personality
```

#### Ollama統合

```bash
# Ollama自動セットアップ（GGUF変換＋登録）
npm run ollama:setup

# GGUFフォーマット変換のみ
npm run ollama:convert

# OllamaAPIテスト
npm run ollama:test

# Ollamaチャット
npm run ollama:chat
```

#### ファイル管理

```bash
# 生成ファイルの削除（outputs/, __pycache__, etc.）
npm run clean
```

#### コマンド詳細説明

| コマンド | 説明 | 実行時間 | 必要条件 |
|---------|------|---------|----------|
| `npm run check-env` | GPU/CUDA環境とパッケージの確認 | 1-2分 | - |
| `npm run train` | LoRA学習（3エポック） | 5-15分 | GPU推奨 |
| `npm run train:quick` | LoRA学習（1エポック） | 2-5分 | GPU推奨 |
| `npm run chat` | リアルタイム対話 | - | 学習済みモデル |
| `npm run test` | 包括的テスト | 3-5分 | 学習済みモデル |
| `npm run ollama:setup` | Ollama統合セットアップ | 5-10分 | Ollama, 学習済みモデル |

#### 実行例

```bash
# 1. 初回セットアップ確認
npm run check-env

# 2. 学習実行
npm run train:quick

# 3. チャットテスト
npm run chat

# 4. モデル評価
npm run test

# 5. Ollama統合（オプション）
npm run ollama:setup
```

### Pythonスクリプト（venv有効化が必要）

**⚠️ 重要: Pythonスクリプトを直接実行する前に、必ず仮想環境を有効化してください**

```bash
# 1. 仮想環境の有効化（必須）
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 2. スクリプトの実行
# 統合実行スクリプト
python run.py help           # ヘルプ表示
python run.py chat           # チャット開始
python run.py train          # 学習実行
python run.py test           # テスト実行

# 個別スクリプト
python chat_interface.py     # チャット
python train_lora.py         # 学習
python test_model.py         # テスト
python inference.py          # 推論
```

### 推奨実行方法

1. **初心者・簡単実行**: `npm run` コマンドを使用
2. **開発・カスタマイズ**: venv有効化後にPythonスクリプトを直接実行

## 主要機能

### 1. チャットインターフェース

```bash
# 対話モード
npm run chat
# または（venv有効化後）
python run.py chat

# デモモード
npm run chat:demo
# または（venv有効化後）
python run.py chat-demo
```

望月京子とリアルタイムで対話できるインターフェース。会話履歴の保存・読み込み機能付き。

### 2. LoRA学習

```bash
# 通常学習（3エポック）
npm run train
# または（venv有効化後）
python run.py train

# 高速学習（1エポック、テスト用）
npm run train:quick
# または（venv有効化後）
python run.py train-quick
```

カスタマイズされた学習データでLoRAファインチューニングを実行。

### 3. モデルテスト

```bash
# 包括的テスト
npm run test
# または（venv有効化後）
python run.py test

# 高速テスト
npm run test:quick
# または（venv有効化後）
python run.py test-quick

# 特定テストのみ
npm run test:basic        # 基本応答テスト
npm run test:personality  # 人格一貫性テスト
```

AITuberの応答品質、人格一貫性、パフォーマンスを評価。

### 4. 推論

```bash
# 対話式推論
npm run inference
# または（venv有効化後）
python run.py inference

# デモ実行
npm run inference:demo
# または（venv有効化後）
python run.py inference-demo
```

学習済みモデルでの単発推論やバッチ推論。

### 5. Ollama統合

```bash
# Ollama統合セットアップ（自動）
bash setup_ollama.sh

# 手動でGGUF変換
python convert_to_gguf.py

# Ollamaでチャット
ollama run aituber-kyoko

# APIテスト
python ollama_test.py
```

学習済みLoRAモデルをOllama（GGUF形式）で高速実行。

## 設定

### 学習設定（config.yaml）

主要なパラメータ：

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"  # ベースモデル
  load_in_8bit: true                   # 8bit量子化

lora:
  r: 16                               # LoRAランク
  lora_alpha: 32                      # スケーリング係数
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  num_train_epochs: 3                 # エポック数
  learning_rate: 2e-4                 # 学習率
  per_device_train_batch_size: 4      # バッチサイズ
```

### モデル変更

より大きなモデルを使用する場合：

```bash
# 学習
python train_lora.py --model "Qwen/Qwen2.5-3B-Instruct"

# チャット
python chat_interface.py --model "Qwen/Qwen2.5-3B-Instruct"
```

## パフォーマンス

- **推奨環境**: NVIDIA GPU（8GB以上のVRAM）
- **最小環境**: CPU（実行は可能だが非常に低速）
- **応答時間**: 通常2-5秒（GPU使用時）

## ファイル管理

```bash
# 生成ファイルの削除
npm run clean
# または
python run.py clean
```

学習済みモデル、テストレポート、一時ファイルを削除。

## Ollama統合

### 前提条件

1. **Ollamaのインストール**:
   ```bash
   # Linux/Mac
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows
   # https://ollama.ai/ からダウンロード
   ```

2. **Ollamaサーバーの起動**:
   ```bash
   ollama serve
   ```

### セットアップ手順

#### 自動セットアップ（推奨）

```bash
# 1. LoRA学習の完了を確認
ls outputs/lora_model/

# 2. 自動セットアップの実行
bash setup_ollama.sh
```

#### 手動セットアップ

```bash
# 1. GGUF変換
python convert_to_gguf.py

# 2. 変換スクリプトの実行
cd ollama_models
bash convert_to_gguf.sh

# 3. Ollamaにモデル登録
ollama create aituber-kyoko -f Modelfile

# 4. テスト実行
ollama run aituber-kyoko "自己紹介をしてください"
```

### 使用方法

#### コマンドライン

```bash
# インタラクティブチャット
ollama run aituber-kyoko

# 単発質問
ollama run aituber-kyoko "今日の調子はどう？"
```

#### API経由

```bash
# 基本的なAPI呼び出し
curl http://localhost:11434/api/generate \
  -d '{"model": "aituber-kyoko", "prompt": "自己紹介をしてください"}'

# ストリーミング応答
curl http://localhost:11434/api/generate \
  -d '{"model": "aituber-kyoko", "prompt": "配信の感想を聞かせて", "stream": true}'
```

#### Python API

```python
import requests

response = requests.post('http://localhost:11434/api/generate', 
    json={
        'model': 'aituber-kyoko',
        'prompt': '今日の配信はどうでしたか？',
        'stream': False
    })

print(response.json()['response'])
```

### パフォーマンス

- **応答速度**: 通常0.5-2秒（GPUなしでも高速）
- **メモリ使用量**: 約2-4GB（量子化レベルによる）
- **CPU使用率**: 最適化済み（llama.cpp）

### 管理コマンド

```bash
# モデル一覧
ollama list

# モデル詳細情報
ollama show aituber-kyoko

# モデル削除
ollama rm aituber-kyoko

# ログ確認
ollama logs
```

## トラブルシューティング

### よくある問題

1. **CUDAエラー**: GPU環境の確認
   ```bash
   python run.py check-env
   ```

2. **メモリ不足**: バッチサイズを削減
   ```yaml
   # config.yaml
   training:
     per_device_train_batch_size: 2  # 4から2に削減
   ```

3. **学習データエラー**: データを再準備
   ```bash
   npm run prepare-data
   ```

4. **Ollama接続エラー**: サーバー起動の確認
   ```bash
   # サーバー起動
   ollama serve
   
   # 接続テスト
   curl http://localhost:11434/api/tags
   ```

5. **GGUF変換エラー**: llama.cppの確認
   ```bash
   # llama.cppのクローン
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp && make
   ```

### デバッグモード

```bash
# デバッグ情報を表示
DEBUG=1 python chat_interface.py
```

## 開発

### データセットの更新

1. `data/training/`内のJSONLファイルを編集
2. データセットを再準備：
   ```bash
   npm run prepare-data
   ```
3. 学習を再実行：
   ```bash
   npm run train
   ```

### 新しいキャラクター設定

`data/character_profile.json`を編集してキャラクター設定を変更可能。

## ライセンス

MIT License

## 技術スタック

- **ベースモデル**: Qwen2.5 (Alibaba)
- **ファインチューニング**: LoRA (Parameter-Efficient Fine-Tuning)
- **ライブラリ**: transformers, peft, datasets, accelerate
- **量子化**: BitsAndBytes (8bit) / GGUF (Ollama)
- **推論エンジン**: Hugging Face Transformers / Ollama (llama.cpp)
- **GPU支援**: CUDA, cuDNN

## 貢献

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
