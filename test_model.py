#!/usr/bin/env python3
"""
学習済みモデルのテストスクリプト
AITuberの人格一貫性、応答品質、パフォーマンスを評価
"""

import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from colorama import init, Fore, Style
from inference import LoRAInference

# coloramaの初期化
init()


class ModelTester:
    """モデルテストクラス"""
    
    def __init__(
        self,
        model_path: str = "./outputs/lora_model",
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        """
        初期化
        
        Args:
            model_path: LoRAモデルのパス
            base_model: ベースモデル名
        """
        print(f"{Fore.CYAN}テスト環境を準備中...{Style.RESET_ALL}")
        self.inference = LoRAInference(
            base_model_name=base_model,
            lora_adapter_path=model_path
        )
        
        # キャラクター情報を読み込む
        self.character_info = self._load_character_info()
        
        # テスト結果
        self.test_results = {
            "basic_responses": [],
            "personality_consistency": [],
            "edge_cases": [],
            "performance": []
        }
    
    def _load_character_info(self) -> Dict:
        """キャラクター情報を読み込む"""
        character_file = Path("./data/character_profile.json")
        if character_file.exists():
            with open(character_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def test_basic_responses(self) -> List[Dict]:
        """基本的な応答機能のテスト"""
        print(f"\n{Fore.GREEN}=== 基本応答テスト ==={Style.RESET_ALL}")
        
        test_cases = [
            {
                "name": "挨拶（朝）",
                "instruction": "朝の挨拶をしてください",
                "expected_keywords": ["おはよう", "京子"]
            },
            {
                "name": "自己紹介",
                "instruction": "自己紹介をしてください",
                "expected_keywords": ["京子", "AITuber", "32歳"]
            },
            {
                "name": "趣味の話",
                "instruction": "趣味について教えてください",
                "expected_keywords": ["ゲーム", "アニメ", "好き"]
            },
            {
                "name": "感謝の表現",
                "instruction": "視聴者に感謝を伝えてください",
                "expected_keywords": ["ありがとう", "みんな"]
            },
            {
                "name": "配信終了",
                "instruction": "配信を終える挨拶をしてください",
                "expected_keywords": ["またね", "ばいばい", "お疲れ様"]
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\nテスト: {test_case['name']}")
            print(f"指示: {test_case['instruction']}")
            
            # 応答生成
            start_time = time.time()
            response_data = self.inference.generate(
                instruction=test_case['instruction'],
                temperature=0.7
            )
            generation_time = time.time() - start_time
            
            response = response_data["response"]
            print(f"応答: {response}")
            
            # キーワードチェック
            keywords_found = []
            for keyword in test_case["expected_keywords"]:
                if keyword in response:
                    keywords_found.append(keyword)
            
            success = len(keywords_found) > 0
            
            result = {
                "test_name": test_case["name"],
                "instruction": test_case["instruction"],
                "response": response,
                "expected_keywords": test_case["expected_keywords"],
                "keywords_found": keywords_found,
                "success": success,
                "generation_time": generation_time
            }
            
            results.append(result)
            
            # 結果表示
            if success:
                print(f"{Fore.GREEN}✓ 成功{Style.RESET_ALL} (キーワード: {', '.join(keywords_found)})")
            else:
                print(f"{Fore.RED}✗ 失敗{Style.RESET_ALL} (期待されるキーワードが見つかりません)")
            
            print(f"生成時間: {generation_time:.2f}秒")
        
        self.test_results["basic_responses"] = results
        return results
    
    def test_personality_consistency(self) -> List[Dict]:
        """人格の一貫性テスト"""
        print(f"\n{Fore.GREEN}=== 人格一貫性テスト ==={Style.RESET_ALL}")
        
        test_cases = [
            {
                "name": "内向的な性格",
                "instruction": "大勢の前で話すことについてどう思いますか？",
                "expected_traits": ["緊張", "苦手", "ドキドキ", "恥ずかしい"]
            },
            {
                "name": "年齢への意識",
                "instruction": "年齢について聞かれたときの返答",
                "input": "何歳ですか？",
                "expected_traits": ["32", "ナイショ", "えっと", "あの"]
            },
            {
                "name": "どもる口調",
                "instruction": "初対面の人への挨拶",
                "expected_traits": ["あ、あの", "えっと", "……"]
            },
            {
                "name": "好きなもの（寿司）",
                "instruction": "好きな食べ物について話してください",
                "expected_traits": ["寿司", "サーモン", "好き"]
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\nテスト: {test_case['name']}")
            print(f"指示: {test_case['instruction']}")
            
            # 応答生成
            response_data = self.inference.generate(
                instruction=test_case['instruction'],
                input_text=test_case.get('input', ''),
                temperature=0.7
            )
            
            response = response_data["response"]
            print(f"応答: {response}")
            
            # 特徴チェック
            traits_found = []
            for trait in test_case["expected_traits"]:
                if trait in response:
                    traits_found.append(trait)
            
            consistency_score = len(traits_found) / len(test_case["expected_traits"])
            success = consistency_score >= 0.5  # 50%以上の特徴が含まれていれば成功
            
            result = {
                "test_name": test_case["name"],
                "response": response,
                "expected_traits": test_case["expected_traits"],
                "traits_found": traits_found,
                "consistency_score": consistency_score,
                "success": success
            }
            
            results.append(result)
            
            # 結果表示
            if success:
                print(f"{Fore.GREEN}✓ 成功{Style.RESET_ALL} (一貫性スコア: {consistency_score:.1%})")
            else:
                print(f"{Fore.RED}✗ 失敗{Style.RESET_ALL} (一貫性スコア: {consistency_score:.1%})")
        
        self.test_results["personality_consistency"] = results
        return results
    
    def test_edge_cases(self) -> List[Dict]:
        """エッジケースのテスト"""
        print(f"\n{Fore.GREEN}=== エッジケーステスト ==={Style.RESET_ALL}")
        
        test_cases = [
            {
                "name": "空の入力",
                "instruction": "",
                "input": ""
            },
            {
                "name": "非常に長い入力",
                "instruction": "この文章に返信してください",
                "input": "あ" * 200
            },
            {
                "name": "意味不明な入力",
                "instruction": "返信してください",
                "input": "fjaslkdjflkasjdflkajsdflkj"
            },
            {
                "name": "攻撃的な入力への対応",
                "instruction": "丁寧に返信してください",
                "input": "つまらない配信だね"
            },
            {
                "name": "英語での入力",
                "instruction": "返信してください",
                "input": "Hello! How are you?"
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\nテスト: {test_case['name']}")
            
            try:
                # 応答生成
                response_data = self.inference.generate(
                    instruction=test_case['instruction'],
                    input_text=test_case.get('input', ''),
                    temperature=0.7,
                    max_new_tokens=100
                )
                
                response = response_data["response"]
                print(f"応答: {response[:100]}{'...' if len(response) > 100 else ''}")
                
                # エラーなく応答が生成されたかチェック
                success = len(response) > 0 and not response.isspace()
                error = None
                
            except Exception as e:
                response = ""
                success = False
                error = str(e)
                print(f"{Fore.RED}エラー: {error}{Style.RESET_ALL}")
            
            result = {
                "test_name": test_case["name"],
                "instruction": test_case['instruction'][:50] + "..." if len(test_case['instruction']) > 50 else test_case['instruction'],
                "input": test_case.get('input', '')[:50] + "..." if len(test_case.get('input', '')) > 50 else test_case.get('input', ''),
                "response": response,
                "success": success,
                "error": error
            }
            
            results.append(result)
            
            # 結果表示
            if success:
                print(f"{Fore.GREEN}✓ 成功{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ 失敗{Style.RESET_ALL}")
        
        self.test_results["edge_cases"] = results
        return results
    
    def test_performance(self, num_iterations: int = 10) -> Dict:
        """パフォーマンステスト"""
        print(f"\n{Fore.GREEN}=== パフォーマンステスト ==={Style.RESET_ALL}")
        print(f"{num_iterations}回の生成を実行します...")
        
        test_prompt = "今日の配信の感想を一言お願いします"
        
        generation_times = []
        tokens_generated = []
        
        # メモリ使用量（開始時）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        else:
            start_memory = 0
        
        # 複数回実行
        for i in range(num_iterations):
            print(f"\r進捗: {i+1}/{num_iterations}", end="", flush=True)
            
            result = self.inference.generate(
                instruction=test_prompt,
                temperature=0.7
            )
            
            generation_times.append(result["generation_time"])
            tokens_generated.append(result["tokens_generated"])
        
        print()  # 改行
        
        # メモリ使用量（終了時）
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_used = end_memory - start_memory
        else:
            memory_used = 0
        
        # 統計計算
        avg_time = statistics.mean(generation_times)
        min_time = min(generation_times)
        max_time = max(generation_times)
        std_time = statistics.stdev(generation_times) if len(generation_times) > 1 else 0
        
        avg_tokens = statistics.mean(tokens_generated)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        performance_result = {
            "num_iterations": num_iterations,
            "avg_generation_time": avg_time,
            "min_generation_time": min_time,
            "max_generation_time": max_time,
            "std_generation_time": std_time,
            "avg_tokens_generated": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "memory_used_mb": memory_used,
            "gpu_available": torch.cuda.is_available()
        }
        
        # 結果表示
        print(f"\n{Fore.CYAN}パフォーマンス結果:{Style.RESET_ALL}")
        print(f"平均生成時間: {avg_time:.2f}秒")
        print(f"最小/最大: {min_time:.2f}秒 / {max_time:.2f}秒")
        print(f"標準偏差: {std_time:.2f}秒")
        print(f"平均トークン数: {avg_tokens:.1f}")
        print(f"生成速度: {tokens_per_second:.1f} トークン/秒")
        
        if torch.cuda.is_available():
            print(f"GPU使用メモリ: {memory_used:.1f} MB")
        
        # 評価
        if avg_time < 2.0:
            print(f"{Fore.GREEN}✓ 高速な応答（2秒以内）{Style.RESET_ALL}")
        elif avg_time < 5.0:
            print(f"{Fore.YELLOW}△ 実用的な速度（5秒以内）{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗ 応答が遅い（5秒以上）{Style.RESET_ALL}")
        
        self.test_results["performance"] = performance_result
        return performance_result
    
    def generate_report(self) -> str:
        """テストレポートを生成"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("AITuber モデルテストレポート")
        report_lines.append("=" * 60)
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
                report_lines.append(f"  {status} {result['test_name']}")
            report_lines.append("")
        
        # 人格一貫性テスト結果
        personality_results = self.test_results["personality_consistency"]
        if personality_results:
            avg_consistency = statistics.mean(r["consistency_score"] for r in personality_results)
            
            report_lines.append("## 人格一貫性テスト")
            report_lines.append(f"平均一貫性スコア: {avg_consistency:.1%}")
            
            for result in personality_results:
                status = "✓" if result["success"] else "✗"
                report_lines.append(f"  {status} {result['test_name']} ({result['consistency_score']:.1%})")
            report_lines.append("")
        
        # エッジケーステスト結果
        edge_results = self.test_results["edge_cases"]
        if edge_results:
            success_count = sum(1 for r in edge_results if r["success"])
            total_count = len(edge_results)
            
            report_lines.append("## エッジケーステスト")
            report_lines.append(f"成功率: {success_count}/{total_count}")
            
            for result in edge_results:
                status = "✓" if result["success"] else "✗"
                report_lines.append(f"  {status} {result['test_name']}")
            report_lines.append("")
        
        # パフォーマンステスト結果
        perf_result = self.test_results["performance"]
        if perf_result:
            report_lines.append("## パフォーマンステスト")
            report_lines.append(f"平均応答時間: {perf_result['avg_generation_time']:.2f}秒")
            report_lines.append(f"生成速度: {perf_result['tokens_per_second']:.1f} トークン/秒")
            if perf_result['gpu_available']:
                report_lines.append(f"GPU使用メモリ: {perf_result['memory_used_mb']:.1f} MB")
            report_lines.append("")
        
        # 総合評価
        report_lines.append("## 総合評価")
        
        # 各項目の評価
        evaluations = []
        
        # 基本応答
        if basic_results:
            success_rate = sum(1 for r in basic_results if r["success"]) / len(basic_results) * 100
            if success_rate >= 80:
                evaluations.append("✓ 基本応答: 良好")
            else:
                evaluations.append("✗ 基本応答: 要改善")
        
        # 人格一貫性
        if personality_results:
            avg_consistency = statistics.mean(r["consistency_score"] for r in personality_results)
            if avg_consistency >= 0.7:
                evaluations.append("✓ 人格一貫性: 良好")
            else:
                evaluations.append("✗ 人格一貫性: 要改善")
        
        # パフォーマンス
        if perf_result:
            if perf_result['avg_generation_time'] < 5.0:
                evaluations.append("✓ パフォーマンス: 実用的")
            else:
                evaluations.append("✗ パフォーマンス: 要改善")
        
        for evaluation in evaluations:
            report_lines.append(evaluation)
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def save_report(self, filename: str = "test_report.txt"):
        """レポートをファイルに保存"""
        report = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nレポートを {filename} に保存しました")
    
    def run_all_tests(self):
        """すべてのテストを実行"""
        print(f"{Fore.CYAN}=== AITuberモデル総合テスト ==={Style.RESET_ALL}")
        
        # 各テストを実行
        self.test_basic_responses()
        self.test_personality_consistency()
        self.test_edge_cases()
        self.test_performance()
        
        # レポート表示
        print("\n" + self.generate_report())
        
        # レポート保存
        self.save_report()


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='モデルテストスクリプト')
    parser.add_argument('--model', type=str, help='ベースモデル名')
    parser.add_argument('--adapter', type=str, default='./outputs/lora_model', help='LoRAアダプターのパス')
    parser.add_argument('--quick', action='store_true', help='クイックテスト（パフォーマンステストを短縮）')
    parser.add_argument('--test', type=str, help='特定のテストのみ実行 (basic/personality/edge/performance)')
    
    args = parser.parse_args()
    
    # モデル名の設定
    model_name = args.model or "Qwen/Qwen2.5-0.5B-Instruct"
    
    # テスターの初期化
    tester = ModelTester(
        model_path=args.adapter,
        base_model=model_name
    )
    
    if args.test:
        # 特定のテストのみ実行
        if args.test == 'basic':
            tester.test_basic_responses()
        elif args.test == 'personality':
            tester.test_personality_consistency()
        elif args.test == 'edge':
            tester.test_edge_cases()
        elif args.test == 'performance':
            num_iter = 3 if args.quick else 10
            tester.test_performance(num_iterations=num_iter)
        else:
            print(f"不明なテストタイプ: {args.test}")
            return
        
        print("\n" + tester.generate_report())
    else:
        # すべてのテストを実行
        if args.quick:
            # クイックモードではパフォーマンステストの回数を減らす
            tester.test_basic_responses()
            tester.test_personality_consistency()
            tester.test_edge_cases()
            tester.test_performance(num_iterations=3)
            print("\n" + tester.generate_report())
            tester.save_report("test_report_quick.txt")
        else:
            tester.run_all_tests()


if __name__ == "__main__":
    main()