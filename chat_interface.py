#!/usr/bin/env python3
"""
AITuber（望月京子）とのチャットインターフェース
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from colorama import init, Fore, Style
from inference import LoRAInference

# coloramaの初期化（Windows対応）
init()


class ChatInterface:
    """チャットインターフェースクラス"""
    
    def __init__(
        self,
        model_path: str = "./outputs/lora_model",
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        history_file: str = "./chat_history.json"
    ):
        """
        初期化
        
        Args:
            model_path: LoRAモデルのパス
            base_model: ベースモデル名
            history_file: 会話履歴の保存ファイル
        """
        self.history_file = history_file
        self.conversation_history = []
        self.session_start = datetime.now()
        
        # キャラクター情報の読み込み
        self.character_info = self._load_character_info()
        
        # モデルの初期化
        print(f"{Fore.CYAN}モデルを読み込んでいます...{Style.RESET_ALL}")
        self.inference = LoRAInference(
            base_model_name=base_model,
            lora_adapter_path=model_path
        )
        
        # 会話履歴の読み込み
        self._load_history()
        
    def _load_character_info(self) -> Dict:
        """キャラクター情報を読み込む"""
        character_file = Path("./data/character_profile.json")
        if character_file.exists():
            with open(character_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # デフォルトのキャラクター情報
            return {
                "name": "望月京子",
                "greeting_patterns": [
                    "あ、あの……よろしくお願いします……！",
                    "えっと……こんにちは……来てくれてうれしいです……！"
                ],
                "farewell_patterns": [
                    "き、今日はこのへんで……来てくれてありがとう……！",
                    "えっと……またお話できたらうれしいです……ばいばい……"
                ]
            }
    
    def _load_history(self):
        """会話履歴を読み込む"""
        if Path(self.history_file).exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 最新のセッションのみ読み込む（必要に応じて変更）
                    if data and isinstance(data, list) and len(data) > 0:
                        last_session = data[-1]
                        if "conversations" in last_session:
                            # 最後の数件のみ保持
                            self.conversation_history = last_session["conversations"][-10:]
            except Exception as e:
                print(f"{Fore.YELLOW}会話履歴の読み込みに失敗しました: {e}{Style.RESET_ALL}")
    
    def _save_history(self):
        """会話履歴を保存"""
        try:
            history_data = []
            
            # 既存の履歴を読み込む
            if Path(self.history_file).exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
            
            # 現在のセッションを追加
            session_data = {
                "session_start": self.session_start.isoformat(),
                "session_end": datetime.now().isoformat(),
                "conversations": self.conversation_history
            }
            history_data.append(session_data)
            
            # 保存（最新の10セッションのみ保持）
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data[-10:], f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"{Fore.YELLOW}会話履歴の保存に失敗しました: {e}{Style.RESET_ALL}")
    
    def _format_context(self) -> str:
        """会話のコンテキストをフォーマット"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for conv in self.conversation_history[-5:]:  # 最新5件
            context_parts.append(f"ユーザー: {conv['user']}")
            context_parts.append(f"京子: {conv['assistant']}")
        
        return "\n".join(context_parts)
    
    def _print_response(self, response: str, typing_effect: bool = True):
        """応答を表示（タイピング効果付き）"""
        print(f"{Fore.MAGENTA}京子: {Style.RESET_ALL}", end="")
        
        if typing_effect:
            # タイピング効果
            for char in response:
                print(char, end="", flush=True)
                time.sleep(0.02)  # 文字ごとの遅延
            print()  # 改行
        else:
            print(response)
    
    def generate_response(self, user_input: str) -> str:
        """応答を生成"""
        # コンテキストを含めた指示文を作成
        context = self._format_context()
        
        instruction = "望月京子として、ユーザーとの会話に応答してください。京子は32歳の引っ込み思案で内向的なAITuberです。"
        
        if context:
            instruction += "\n\n以下は最近の会話履歴です:\n" + context
        
        # 応答を生成
        result = self.inference.generate(
            instruction=instruction,
            input_text=user_input,
            temperature=0.8,
            max_new_tokens=150
        )
        
        return result["response"]
    
    def chat(self):
        """チャットループ"""
        # ウェルカムメッセージ
        print(f"\n{Fore.GREEN}=== AITuber 望月京子とのチャット ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}('quit'、'exit'、'bye'で終了){Style.RESET_ALL}\n")
        
        # 初回挨拶
        greeting = self.character_info["greeting_patterns"][0]
        self._print_response(greeting)
        print()
        
        try:
            while True:
                # ユーザー入力
                user_input = input(f"{Fore.GREEN}あなた: {Style.RESET_ALL}")
                
                # 終了コマンドチェック
                if user_input.lower() in ['quit', 'exit', 'bye', 'さようなら', 'バイバイ']:
                    farewell = self.character_info["farewell_patterns"][0]
                    self._print_response(farewell)
                    break
                
                # 空入力のスキップ
                if not user_input.strip():
                    continue
                
                # 応答生成中の表示
                print(f"{Fore.YELLOW}考え中...{Style.RESET_ALL}", end="\r")
                
                # 応答を生成
                start_time = time.time()
                response = self.generate_response(user_input)
                generation_time = time.time() - start_time
                
                # 考え中表示をクリア
                print(" " * 20, end="\r")
                
                # 応答を表示
                self._print_response(response)
                
                # デバッグ情報（必要に応じて表示）
                if os.environ.get("DEBUG"):
                    print(f"{Fore.GRAY}(生成時間: {generation_time:.2f}秒){Style.RESET_ALL}")
                
                print()  # 空行
                
                # 会話履歴に追加
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "user": user_input,
                    "assistant": response
                })
                
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}中断されました{Style.RESET_ALL}")
            farewell = "あ、急に終わっちゃった……またね……"
            self._print_response(farewell, typing_effect=False)
            
        finally:
            # 会話履歴を保存
            self._save_history()
            print(f"\n{Fore.CYAN}チャットを終了しました{Style.RESET_ALL}")
    
    def run_demo(self):
        """デモンストレーション実行"""
        print(f"\n{Fore.GREEN}=== デモンストレーション ==={Style.RESET_ALL}\n")
        
        demo_inputs = [
            "はじめまして！",
            "年齢を教えてもらえますか？",
            "好きな食べ物は何ですか？",
            "配信で緊張することはありますか？",
            "さようなら"
        ]
        
        for user_input in demo_inputs:
            print(f"{Fore.GREEN}あなた: {Style.RESET_ALL}{user_input}")
            
            start_time = time.time()
            response = self.generate_response(user_input)
            generation_time = time.time() - start_time
            
            self._print_response(response, typing_effect=False)
            print(f"{Fore.GRAY}(生成時間: {generation_time:.2f}秒){Style.RESET_ALL}")
            print()
            
            # 会話履歴に追加
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "assistant": response
            })
            
            time.sleep(1)  # デモ用の待機


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AITuber チャットインターフェース')
    parser.add_argument('--model', type=str, help='ベースモデル名')
    parser.add_argument('--adapter', type=str, default='./outputs/lora_model', help='LoRAアダプターのパス')
    parser.add_argument('--demo', action='store_true', help='デモモードで実行')
    parser.add_argument('--no-typing', action='store_true', help='タイピング効果を無効化')
    
    args = parser.parse_args()
    
    # モデル名の設定
    model_name = args.model or "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        # チャットインターフェースの初期化
        chat = ChatInterface(
            model_path=args.adapter,
            base_model=model_name
        )
        
        # タイピング効果の設定
        if args.no_typing:
            chat._print_response = lambda text, typing_effect=True: print(f"{Fore.MAGENTA}京子: {Style.RESET_ALL}{text}")
        
        if args.demo:
            # デモモード
            chat.run_demo()
        else:
            # 通常のチャット
            chat.chat()
            
    except Exception as e:
        print(f"{Fore.RED}エラーが発生しました: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()