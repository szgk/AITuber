#!/usr/bin/env python3
"""
学習データセットの準備と検証スクリプト
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

def load_jsonl(file_path: Path) -> List[Dict]:
    """JSONLファイルを読み込む"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error in {file_path}: {e}")
                print(f"Problem line: {line}")
    return data

def validate_data(data: List[Dict]) -> Tuple[bool, List[str]]:
    """データの形式を検証"""
    errors = []
    required_fields = ['instruction', 'input', 'output']
    
    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                errors.append(f"Entry {i}: Missing field '{field}'")
        
        # outputが空でないことを確認
        if 'output' in item and not item['output'].strip():
            errors.append(f"Entry {i}: Empty output field")
    
    return len(errors) == 0, errors

def merge_and_shuffle(datasets: List[List[Dict]]) -> List[Dict]:
    """複数のデータセットをマージしてシャッフル"""
    merged = []
    for dataset in datasets:
        merged.extend(dataset)
    
    random.shuffle(merged)
    return merged

def split_dataset(data: List[Dict], train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
    """データセットを訓練用と検証用に分割"""
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

def save_jsonl(data: List[Dict], file_path: Path):
    """データをJSONL形式で保存"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    """メイン処理"""
    # データディレクトリ
    data_dir = Path(__file__).parent
    
    # 各データファイルを読み込み
    print("データファイルを読み込み中...")
    greetings = load_jsonl(data_dir / "greetings_and_introductions.jsonl")
    questions = load_jsonl(data_dir / "common_questions.jsonl")
    interactions = load_jsonl(data_dir / "streaming_interactions.jsonl")
    
    print(f"挨拶・自己紹介: {len(greetings)}件")
    print(f"よくある質問: {len(questions)}件")
    print(f"配信中の対話: {len(interactions)}件")
    
    # データの検証
    print("\nデータを検証中...")
    all_valid = True
    for name, dataset in [
        ("greetings", greetings),
        ("questions", questions),
        ("interactions", interactions)
    ]:
        valid, errors = validate_data(dataset)
        if not valid:
            print(f"{name}のエラー:")
            for error in errors[:5]:  # 最初の5件のみ表示
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
            all_valid = False
    
    if not all_valid:
        print("\nエラーが見つかりました。修正してください。")
        return
    
    print("すべてのデータが検証を通過しました。")
    
    # データをマージしてシャッフル
    print("\nデータをマージしてシャッフル中...")
    all_data = merge_and_shuffle([greetings, questions, interactions])
    print(f"合計: {len(all_data)}件")
    
    # 訓練用と検証用に分割
    print("\nデータを分割中...")
    train_data, val_data = split_dataset(all_data, train_ratio=0.9)
    print(f"訓練用: {len(train_data)}件")
    print(f"検証用: {len(val_data)}件")
    
    # ファイルに保存
    print("\nファイルに保存中...")
    save_jsonl(train_data, data_dir / "train.jsonl")
    save_jsonl(val_data, data_dir / "validation.jsonl")
    save_jsonl(all_data, data_dir / "all_data.jsonl")
    
    print("\n完了！以下のファイルが作成されました:")
    print(f"  - train.jsonl ({len(train_data)}件)")
    print(f"  - validation.jsonl ({len(val_data)}件)")
    print(f"  - all_data.jsonl ({len(all_data)}件)")
    
    # 統計情報
    print("\n=== データセット統計 ===")
    print(f"総データ数: {len(all_data)}件")
    print(f"平均instruction長: {sum(len(d['instruction']) for d in all_data) / len(all_data):.1f}文字")
    print(f"平均output長: {sum(len(d['output']) for d in all_data) / len(all_data):.1f}文字")

if __name__ == "__main__":
    random.seed(42)  # 再現性のためのシード
    main()