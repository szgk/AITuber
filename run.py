#!/usr/bin/env python3
"""
AITuber実行スクリプト
簡単なコマンドでモデルの学習・チャット・テストを実行
"""

import sys
import argparse
import subprocess
from pathlib import Path

def run_command(command, description):
    """コマンドを実行"""
    print(f"\n🚀 {description}")
    print(f"実行中: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"\n✅ {description}が完了しました")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ エラーが発生しました: {e}")
        return False

def check_environment():
    """環境チェック"""
    print("🔍 環境をチェックしています...")
    
    # venvの存在確認
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ 仮想環境が見つかりません。セットアップを実行してください: python run.py setup")
        return False
    
    # Pythonの確認
    try:
        result = subprocess.run("python --version", shell=True, capture_output=True, text=True)
        print(f"✅ Python: {result.stdout.strip()}")
    except:
        print("❌ Pythonが見つかりません")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='AITuber実行スクリプト')
    parser.add_argument('command', help='実行するコマンド', choices=[
        'setup', 'train', 'train-quick', 'chat', 'chat-demo', 
        'test', 'test-quick', 'inference', 'inference-demo',
        'prepare-data', 'check-env', 'clean', 'help'
    ])
    parser.add_argument('--no-venv', action='store_true', help='仮想環境を使用しない')
    
    args = parser.parse_args()
    
    # ヘルプ表示
    if args.command == 'help':
        print("AITuber 望月京子 実行スクリプト")
        print("=" * 40)
        print("利用可能なコマンド:")
        print("  setup        - 環境セットアップ")
        print("  train        - LoRA学習の実行")
        print("  train-quick  - 高速学習（1エポック）")
        print("  chat         - チャットインターフェース")
        print("  chat-demo    - チャットデモ")
        print("  test         - 包括的なモデルテスト")
        print("  test-quick   - 高速テスト")
        print("  inference    - 推論モード（対話）")
        print("  inference-demo - 推論デモ")
        print("  prepare-data - データ準備")
        print("  check-env    - 環境チェック")
        print("  clean        - 生成ファイルの削除")
        print("  help         - このヘルプを表示")
        print("\n使用例:")
        print("  python run.py setup")
        print("  python run.py chat")
        print("  python run.py train")
        return
    
    # 仮想環境のプレフィックス
    if args.no_venv:
        python_cmd = "python"
    else:
        if sys.platform == "win32":
            python_cmd = "venv\\Scripts\\python"
        else:
            python_cmd = "source venv/bin/activate && python"
    
    # コマンド実行
    commands = {
        'setup': {
            'cmd': 'python -m venv venv && echo "仮想環境を作成しました。以下を実行してください:" && echo "1. 仮想環境の有効化" && echo "2. pip install -r requirements.txt"',
            'desc': '環境セットアップ'
        },
        'train': {
            'cmd': f'{python_cmd} train_lora.py --config config.yaml',
            'desc': 'LoRA学習の実行'
        },
        'train-quick': {
            'cmd': f'{python_cmd} train_lora.py --config config.yaml --epochs 1',
            'desc': '高速学習（1エポック）'
        },
        'chat': {
            'cmd': f'{python_cmd} chat_interface.py',
            'desc': 'チャットインターフェース'
        },
        'chat-demo': {
            'cmd': f'{python_cmd} chat_interface.py --demo',
            'desc': 'チャットデモ'
        },
        'test': {
            'cmd': f'{python_cmd} test_model.py',
            'desc': '包括的なモデルテスト'
        },
        'test-quick': {
            'cmd': f'{python_cmd} test_model.py --quick',
            'desc': '高速テスト'
        },
        'inference': {
            'cmd': f'{python_cmd} inference.py --interactive',
            'desc': '推論モード（対話）'
        },
        'inference-demo': {
            'cmd': f'{python_cmd} inference.py',
            'desc': '推論デモ'
        },
        'prepare-data': {
            'cmd': f'cd data/training && {python_cmd} prepare_dataset.py',
            'desc': 'データ準備'
        },
        'check-env': {
            'cmd': f'{python_cmd} test_lora_setup.py',
            'desc': '環境チェック'
        },
        'clean': {
            'cmd': 'rm -rf outputs/ __pycache__/ *.pyc .pytest_cache/ test_report*.txt chat_history.json',
            'desc': '生成ファイルの削除'
        }
    }
    
    if args.command in commands:
        # セットアップ以外は環境チェックを実行
        if args.command != 'setup' and args.command != 'clean' and not args.no_venv:
            if not check_environment():
                return
        
        cmd_info = commands[args.command]
        success = run_command(cmd_info['cmd'], cmd_info['desc'])
        
        if not success:
            print("\n❌ コマンドの実行に失敗しました")
            sys.exit(1)
    else:
        print(f"❌ 不明なコマンド: {args.command}")
        print("利用可能なコマンドを確認するには: python run.py help")
        sys.exit(1)

if __name__ == "__main__":
    main()