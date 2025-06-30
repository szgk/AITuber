#!/usr/bin/env python3
"""
LoRA学習環境のセットアップテスト
必要なライブラリと設定を確認する
"""

import sys
from pathlib import Path

def test_imports():
    """必要なライブラリのインポートをテスト"""
    print("=== ライブラリインポートテスト ===")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA利用可能: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        return False
    
    try:
        import peft
        print(f"✓ PEFT: {peft.__version__}")
    except ImportError as e:
        print(f"✗ PEFT: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"✗ Datasets: {e}")
        return False
    
    try:
        import accelerate
        print(f"✓ Accelerate: {accelerate.__version__}")
    except ImportError as e:
        print(f"✗ Accelerate: {e}")
        return False
    
    try:
        import bitsandbytes
        print(f"✓ BitsAndBytes: {bitsandbytes.__version__}")
    except ImportError as e:
        print(f"✗ BitsAndBytes: {e}")
        return False
    
    try:
        import yaml
        print(f"✓ PyYAML: インポート成功")
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        return False
    
    return True

def test_data_files():
    """データファイルの存在を確認"""
    print("\n=== データファイル確認 ===")
    
    data_files = [
        Path("./data/training/train.jsonl"),
        Path("./data/training/validation.jsonl"),
        Path("./config.yaml")
    ]
    
    all_exist = True
    for file_path in data_files:
        if file_path.exists():
            print(f"✓ {file_path}")
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    print(f"  データ数: {len(lines)}件")
        else:
            print(f"✗ {file_path} が見つかりません")
            all_exist = False
    
    return all_exist

def test_lora_config():
    """LoRA設定を簡単にテスト"""
    print("\n=== LoRA設定テスト ===")
    
    try:
        from peft import LoraConfig, TaskType
        
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        print("✓ LoRA設定の作成成功")
        print(f"  r: {config.r}")
        print(f"  lora_alpha: {config.lora_alpha}")
        return True
    except Exception as e:
        print(f"✗ LoRA設定エラー: {e}")
        return False

def main():
    """メインテスト関数"""
    print("LoRA学習環境のセットアップをテストします...\n")
    
    tests = [
        ("ライブラリインポート", test_imports),
        ("データファイル", test_data_files),
        ("LoRA設定", test_lora_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n=== テスト結果まとめ ===")
    all_passed = True
    for test_name, result in results:
        status = "✓ 成功" if result else "✗ 失敗"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\nすべてのテストに成功しました！")
        print("train_lora.pyを実行できます。")
        print("\n実行例:")
        print("  python train_lora.py --config config.yaml")
        print("  python train_lora.py --epochs 1  # テスト用に1エポックのみ")
    else:
        print("\nいくつかのテストが失敗しました。")
        print("エラーを修正してから再度実行してください。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)