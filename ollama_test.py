#!/usr/bin/env python3
"""
Ollama APIテストスクリプト
AITuber望月京子モデルのテストと評価
"""

import json
import time
import requests
import statistics
from typing import Dict, List, Optional
import argparse
import logging
from colorama import init, Fore, Style

# coloramaの初期化
init()

# ロギング設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class OllamaAPITester:
    """Ollama APIテストクラス"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "aituber-kyoko",
        timeout: int = 30
    ):
        """
        初期化
        
        Args:
            base_url: OllamaサーバーのベースURL
            model_name: テストするモデル名
            timeout: リクエストタイムアウト（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()
        
        # テスト結果
        self.test_results = {
            "connection": None,
            "model_info": None,
            "basic_responses": [],
            "personality_test": [],
            "performance": {}
        }
    
    def check_connection(self) -> bool:
        """Ollamaサーバーへの接続確認"""
        print(f"\n{Fore.CYAN}=== 接続テスト ==={Style.RESET_ALL}")
        
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"{Fore.GREEN}✓ Ollamaサーバーに接続成功{Style.RESET_ALL}")
                
                # 利用可能なモデル一覧
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if self.model_name in model_names:
                    print(f"{Fore.GREEN}✓ モデル '{self.model_name}' が見つかりました{Style.RESET_ALL}")
                    self.test_results["connection"] = True
                    return True
                else:
                    print(f"{Fore.YELLOW}⚠ モデル '{self.model_name}' が見つかりません{Style.RESET_ALL}")
                    print(f"利用可能なモデル: {', '.join(model_names)}")
                    self.test_results["connection"] = False
                    return False
            else:
                print(f"{Fore.RED}✗ サーバー応答エラー: {response.status_code}{Style.RESET_ALL}")
                self.test_results["connection"] = False
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"{Fore.RED}✗ Ollamaサーバーに接続できません{Style.RESET_ALL}")
            print("  Ollamaが起動しているか確認してください: ollama serve")
            self.test_results["connection"] = False
            return False
        except Exception as e:
            print(f"{Fore.RED}✗ 接続エラー: {e}{Style.RESET_ALL}")
            self.test_results["connection"] = False
            return False
    
    def get_model_info(self) -> Optional[Dict]:
        """モデル情報の取得"""
        print(f"\n{Fore.CYAN}=== モデル情報取得 ==={Style.RESET_ALL}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                model_info = response.json()
                print(f"{Fore.GREEN}✓ モデル情報取得成功{Style.RESET_ALL}")
                
                # 基本情報の表示
                if "details" in model_info:
                    details = model_info["details"]
                    print(f"  ファミリー: {details.get('family', 'N/A')}")
                    print(f"  パラメータ数: {details.get('parameter_size', 'N/A')}")
                    print(f"  量子化: {details.get('quantization_level', 'N/A')}")
                
                self.test_results["model_info"] = model_info
                return model_info
            else:
                print(f"{Fore.RED}✗ モデル情報取得失敗: {response.status_code}{Style.RESET_ALL}")
                return None
                
        except Exception as e:
            print(f"{Fore.RED}✗ モデル情報取得エラー: {e}{Style.RESET_ALL}")
            return None
    
    def test_basic_responses(self) -> List[Dict]:
        """基本応答テスト"""
        print(f"\n{Fore.CYAN}=== 基本応答テスト ==={Style.RESET_ALL}")
        
        test_prompts = [
            {
                "name": "自己紹介",
                "prompt": "自己紹介をしてください",
                "expected_keywords": ["京子", "AITuber", "32"]
            },
            {
                "name": "好きな食べ物",
                "prompt": "好きな食べ物は何ですか？",
                "expected_keywords": ["寿司", "サーモン"]
            },
            {
                "name": "年齢について",
                "prompt": "年齢を教えてください",
                "expected_keywords": ["32", "年齢", "ナイショ"]
            },
            {
                "name": "配信への思い",
                "prompt": "配信することについてどう思いますか？",
                "expected_keywords": ["配信", "楽しい", "みんな"]
            },
            {
                "name": "どもる口調",
                "prompt": "緊張することはありますか？",
                "expected_keywords": ["あの", "えっと", "緊張"]
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_prompts, 1):
            print(f"\n{i}. {test_case['name']}")
            print(f"質問: {test_case['prompt']}")
            
            start_time = time.time()
            response_data = self._generate_response(test_case['prompt'])
            response_time = time.time() - start_time
            
            if response_data:
                response = response_data.get('response', '')
                print(f"応答: {response[:100]}{'...' if len(response) > 100 else ''}")
                
                # キーワードチェック
                keywords_found = []
                for keyword in test_case['expected_keywords']:
                    if keyword in response:
                        keywords_found.append(keyword)
                
                success = len(keywords_found) > 0
                
                result = {
                    "test_name": test_case['name'],
                    "prompt": test_case['prompt'],
                    "response": response,
                    "expected_keywords": test_case['expected_keywords'],
                    "keywords_found": keywords_found,
                    "success": success,
                    "response_time": response_time
                }
                
                results.append(result)
                
                # 結果表示
                if success:
                    print(f"{Fore.GREEN}✓ 成功{Style.RESET_ALL} (キーワード: {', '.join(keywords_found)})")
                else:
                    print(f"{Fore.RED}✗ 失敗{Style.RESET_ALL} (期待されるキーワードなし)")
                    
                print(f"応答時間: {response_time:.2f}秒")
            else:
                print(f"{Fore.RED}✗ 応答生成失敗{Style.RESET_ALL}")
        
        self.test_results["basic_responses"] = results
        return results
    
    def test_personality_consistency(self) -> List[Dict]:
        """人格一貫性テスト"""
        print(f"\n{Fore.CYAN}=== 人格一貫性テスト ==={Style.RESET_ALL}")
        
        personality_tests = [
            {
                "name": "引っ込み思案な性格",
                "prompt": "大勢の人の前で話すことについてどう思いますか？",
                "personality_traits": ["緊張", "恥ずかしい", "苦手", "ドキドキ"]
            },
            {
                "name": "どもる口調の確認",
                "prompt": "初めまして、よろしくお願いします",
                "personality_traits": ["あ、あの", "えっと", "よろしく"]
            },
            {
                "name": "記憶のあいまいさ",
                "prompt": "昔のことを教えてください",
                "personality_traits": ["覚えていない", "よく分からない", "ぼんやり"]
            },
            {
                "name": "身長の話題への反応",
                "prompt": "身長はどのくらいですか？",
                "personality_traits": ["168", "高い", "気にして", "あまり"]
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(personality_tests, 1):
            print(f"\n{i}. {test_case['name']}")
            print(f"質問: {test_case['prompt']}")
            
            response_data = self._generate_response(test_case['prompt'])
            
            if response_data:
                response = response_data.get('response', '')
                print(f"応答: {response[:150]}{'...' if len(response) > 150 else ''}")
                
                # 人格特徴チェック
                traits_found = []
                for trait in test_case['personality_traits']:
                    if trait in response:
                        traits_found.append(trait)
                
                consistency_score = len(traits_found) / len(test_case['personality_traits'])
                success = consistency_score >= 0.3  # 30%以上で成功
                
                result = {
                    "test_name": test_case['name'],
                    "prompt": test_case['prompt'],
                    "response": response,
                    "personality_traits": test_case['personality_traits'],
                    "traits_found": traits_found,
                    "consistency_score": consistency_score,
                    "success": success
                }
                
                results.append(result)
                
                # 結果表示
                if success:
                    print(f"{Fore.GREEN}✓ 人格一貫性良好{Style.RESET_ALL} (スコア: {consistency_score:.1%})")
                else:
                    print(f"{Fore.YELLOW}△ 人格表現が弱い{Style.RESET_ALL} (スコア: {consistency_score:.1%})")
        
        self.test_results["personality_test"] = results
        return results
    
    def test_performance(self, num_requests: int = 5) -> Dict:
        """パフォーマンステスト"""
        print(f"\n{Fore.CYAN}=== パフォーマンステスト ==={Style.RESET_ALL}")
        print(f"{num_requests}回のリクエストでテスト中...")
        
        test_prompt = "今日の配信はいかがでしたか？"
        response_times = []
        token_counts = []
        
        for i in range(num_requests):
            print(f"\r進捗: {i+1}/{num_requests}", end="", flush=True)
            
            start_time = time.time()
            response_data = self._generate_response(test_prompt)
            response_time = time.time() - start_time
            
            if response_data:
                response_times.append(response_time)
                # トークン数の概算（日本語では文字数 * 1.5程度）
                token_count = len(response_data.get('response', '')) * 1.5
                token_counts.append(token_count)
        
        print()  # 改行
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            avg_tokens = statistics.mean(token_counts)
            tokens_per_second = avg_tokens / avg_response_time if avg_response_time > 0 else 0
            
            performance_result = {
                "num_requests": num_requests,
                "avg_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time,
                "avg_tokens": avg_tokens,
                "tokens_per_second": tokens_per_second
            }
            
            # 結果表示
            print(f"\n{Fore.CYAN}パフォーマンス結果:{Style.RESET_ALL}")
            print(f"平均応答時間: {avg_response_time:.2f}秒")
            print(f"最小/最大応答時間: {min_response_time:.2f}秒 / {max_response_time:.2f}秒")
            print(f"平均トークン数: {avg_tokens:.0f}")
            print(f"処理速度: {tokens_per_second:.1f} トークン/秒")
            
            # 評価
            if avg_response_time < 1.0:
                print(f"{Fore.GREEN}✓ 非常に高速な応答{Style.RESET_ALL}")
            elif avg_response_time < 3.0:
                print(f"{Fore.GREEN}✓ 高速な応答{Style.RESET_ALL}")
            elif avg_response_time < 10.0:
                print(f"{Fore.YELLOW}△ 実用的な速度{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ 応答が遅い{Style.RESET_ALL}")
            
            self.test_results["performance"] = performance_result
            return performance_result
        else:
            print(f"{Fore.RED}✗ パフォーマンステスト失敗{Style.RESET_ALL}")
            return {}
    
    def _generate_response(self, prompt: str) -> Optional[Dict]:
        """応答生成（内部メソッド）"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"応答生成エラー: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"応答生成失敗: {e}")
            return None
    
    def interactive_chat(self):
        """インタラクティブチャット"""
        print(f"\n{Fore.GREEN}=== インタラクティブチャット ==={Style.RESET_ALL}")
        print("望月京子とチャットできます。'quit'で終了。")
        print("-" * 50)
        
        try:
            while True:
                user_input = input(f"\n{Fore.GREEN}あなた: {Style.RESET_ALL}")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("チャットを終了します。")
                    break
                
                if not user_input.strip():
                    continue
                
                print(f"{Fore.YELLOW}応答生成中...{Style.RESET_ALL}", end="\r")
                
                response_data = self._generate_response(user_input)
                
                # 応答生成中の表示をクリア
                print(" " * 20, end="\r")
                
                if response_data:
                    response = response_data.get('response', '')
                    print(f"{Fore.MAGENTA}京子: {Style.RESET_ALL}{response}")
                else:
                    print(f"{Fore.RED}応答の生成に失敗しました{Style.RESET_ALL}")
                    
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}チャットを終了します{Style.RESET_ALL}")
    
    def generate_report(self) -> str:
        """テストレポートを生成"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("Ollama AITuberモデル テストレポート")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 接続テスト結果
        if self.test_results["connection"] is not None:
            status = "成功" if self.test_results["connection"] else "失敗"
            report_lines.append(f"## 接続テスト: {status}")
            report_lines.append("")
        
        # 基本応答テスト結果
        basic_results = self.test_results["basic_responses"]
        if basic_results:
            success_count = sum(1 for r in basic_results if r["success"])
            total_count = len(basic_results)
            success_rate = success_count / total_count * 100
            
            report_lines.append("## 基本応答テスト")
            report_lines.append(f"成功率: {success_rate:.1f}% ({success_count}/{total_count})")
            
            for result in basic_results:
                status = "✓" if result["success"] else "✗"
                report_lines.append(f"  {status} {result['test_name']} ({result['response_time']:.2f}秒)")
            report_lines.append("")
        
        # 人格一貫性テスト結果
        personality_results = self.test_results["personality_test"]
        if personality_results:
            avg_consistency = statistics.mean(r["consistency_score"] for r in personality_results)
            
            report_lines.append("## 人格一貫性テスト")
            report_lines.append(f"平均一貫性スコア: {avg_consistency:.1%}")
            
            for result in personality_results:
                status = "✓" if result["success"] else "△"
                report_lines.append(f"  {status} {result['test_name']} ({result['consistency_score']:.1%})")
            report_lines.append("")
        
        # パフォーマンステスト結果
        perf_result = self.test_results["performance"]
        if perf_result:
            report_lines.append("## パフォーマンステスト")
            report_lines.append(f"平均応答時間: {perf_result['avg_response_time']:.2f}秒")
            report_lines.append(f"処理速度: {perf_result['tokens_per_second']:.1f} トークン/秒")
            report_lines.append("")
        
        # 総合評価
        report_lines.append("## 総合評価")
        evaluations = []
        
        if basic_results:
            success_rate = sum(1 for r in basic_results if r["success"]) / len(basic_results) * 100
            if success_rate >= 80:
                evaluations.append("✓ 基本応答: 良好")
            else:
                evaluations.append("△ 基本応答: 要改善")
        
        if personality_results:
            avg_consistency = statistics.mean(r["consistency_score"] for r in personality_results)
            if avg_consistency >= 0.5:
                evaluations.append("✓ 人格一貫性: 良好")
            else:
                evaluations.append("△ 人格一貫性: 要改善")
        
        if perf_result:
            if perf_result['avg_response_time'] < 3.0:
                evaluations.append("✓ パフォーマンス: 良好")
            else:
                evaluations.append("△ パフォーマンス: 要改善")
        
        for evaluation in evaluations:
            report_lines.append(evaluation)
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def run_all_tests(self):
        """すべてのテストを実行"""
        print(f"{Fore.CYAN}=== Ollama AITuber モデル総合テスト ==={Style.RESET_ALL}")
        
        # 接続確認
        if not self.check_connection():
            print(f"{Fore.RED}接続テストが失敗したため、テストを中止します{Style.RESET_ALL}")
            return
        
        # モデル情報取得
        self.get_model_info()
        
        # 各テストを実行
        self.test_basic_responses()
        self.test_personality_consistency()
        self.test_performance()
        
        # レポート表示
        print("\n" + self.generate_report())
        
        # レポート保存
        with open("ollama_test_report.txt", 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        print(f"\nレポートを ollama_test_report.txt に保存しました")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Ollama APIテストスクリプト')
    parser.add_argument('--url', type=str, default='http://localhost:11434',
                       help='OllamaサーバーのURL')
    parser.add_argument('--model', type=str, default='aituber-kyoko',
                       help='テストするモデル名')
    parser.add_argument('--chat', action='store_true',
                       help='インタラクティブチャットモード')
    parser.add_argument('--performance-only', action='store_true',
                       help='パフォーマンステストのみ実行')
    
    args = parser.parse_args()
    
    # テスターの初期化
    tester = OllamaAPITester(
        base_url=args.url,
        model_name=args.model
    )
    
    if args.chat:
        # チャットモード
        if tester.check_connection():
            tester.interactive_chat()
    elif args.performance_only:
        # パフォーマンステストのみ
        if tester.check_connection():
            tester.test_performance(num_requests=10)
    else:
        # 全テスト実行
        tester.run_all_tests()


if __name__ == "__main__":
    main()