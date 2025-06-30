# AITuber 望月京子

LoRA（Low-Rank Adaptation）を使用したAITuberシステム

## 概要

このプロジェクトは、Qwen2.5ベースモデルにLoRAファインチューニングを適用して、「望月京子」というAITuberキャラクターを作成するシステムです。

## キャラクター設定

**望月京子（もちづき きょうこ）**
- 年齢: 32歳
- 性格: 引っ込み思案で内向的、どこか自信なさげ
- 特徴: どもる口調（「あ、あの……」「えっと……」）
- 好きなもの: 寿司（特にサーモン）、アニメ・マンガ、小さくてかわいい雑貨

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
python -m venv venv

# 仮想環境の有効化
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# パッケージのインストール
pip install -r requirements.txt
```

### 2. 環境確認

```bash
# 環境テスト
python test_lora_setup.py

# または
python run.py check-env
```

## 使用方法

### npmスクリプト（推奨）

```bash
# 利用可能なコマンドを確認
npm run help

# 主要なコマンド
npm run chat          # チャットインターフェース
npm run train         # LoRA学習の実行
npm run test          # モデルテスト
npm run inference     # 推論モード
```

### Pythonスクリプト

```bash
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

## 主要機能

### 1. チャットインターフェース

```bash
# 対話モード
npm run chat
# または
python run.py chat

# デモモード
npm run chat:demo
# または
python run.py chat-demo
```

望月京子とリアルタイムで対話できるインターフェース。会話履歴の保存・読み込み機能付き。

### 2. LoRA学習

```bash
# 通常学習（3エポック）
npm run train
# または
python run.py train

# 高速学習（1エポック、テスト用）
npm run train:quick
# または
python run.py train-quick
```

カスタマイズされた学習データでLoRAファインチューニングを実行。

### 3. モデルテスト

```bash
# 包括的テスト
npm run test
# または
python run.py test

# 高速テスト
npm run test:quick
# または
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
# または
python run.py inference

# デモ実行
npm run inference:demo
# または
python run.py inference-demo
```

学習済みモデルでの単発推論やバッチ推論。

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
- **量子化**: BitsAndBytes (8bit)
- **GPU支援**: CUDA, cuDNN

## 貢献

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request