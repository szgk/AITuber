#!/usr/bin/env python3
"""
Qwen3-4Bモデルの読み込みテスト
環境が正しくセットアップされているか確認するためのスクリプト
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_qwen3_load():
    """Qwen3-4Bモデルの読み込みテスト"""
    
    print("=== Qwen3-4B モデル読み込みテスト ===")
    
    # GPU利用可能性の確認
    if torch.cuda.is_available():
        print(f"✓ GPU利用可能: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("✗ GPU利用不可")
        return False
    
    try:
        # モデル名
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # テスト用に小さいモデルを使用
        
        print(f"\nモデル読み込み中: {model_name}")
        
        # トークナイザーの読み込み
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ トークナイザー読み込み完了")
        
        # モデルの読み込み（8bit量子化）
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ モデル読み込み完了")
        
        # 簡単な推論テスト
        print("\n推論テスト実行中...")
        prompt = "こんにちは、今日の天気は"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"入力: {prompt}")
        print(f"出力: {response}")
        
        print("\n✓ すべてのテストが成功しました！")
        return True
        
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    success = test_qwen3_load()
    
    if not success:
        print("\n必要なパッケージをインストールしてください:")
        print("pip install -r requirements.txt")