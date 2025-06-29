# LoRA学習環境セットアップガイド

## 環境確認結果

✅ **Python環境**: Python 3.12.3
✅ **GPU環境**: Quadro RTX 4000 (8GB VRAM)
✅ **CUDA環境**: CUDA 12.0

## セットアップ手順

### 1. 必要なパッケージのインストール

```bash
# pip/pip3が利用可能な環境で実行
pip install -r requirements.txt
```

または個別にインストール:

```bash
pip install transformers datasets peft accelerate bitsandbytes torch
```

### 2. 環境の確認

セットアップが正しく完了したか確認:

```bash
python3 test_qwen3_load.py
```

## トラブルシューティング

### CUDA関連のエラーが出る場合

1. PyTorchのCUDAバージョンを確認:
```bash
python3 -c "import torch; print(torch.version.cuda)"
```

2. 必要に応じてPyTorchを再インストール:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
```

### メモリ不足エラーが出る場合

- 8bit量子化を使用（サンプルコードで既に設定済み）
- バッチサイズを小さくする
- より小さいモデル（Qwen2.5-0.5B）でテスト

## 次のステップ

環境構築が完了したら、以下のissueに進んでください:

1. #6 AITuber用学習データの準備
2. #7 LoRA学習スクリプトの実装
3. #8 学習済みモデルの推論・テスト実装
4. #9 OllamaへのLoRAモデル統合