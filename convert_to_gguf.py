#!/usr/bin/env python3
"""
LoRAモデルをGGUF形式に変換するスクリプト
Ollama用にHugging Faceモデルを変換
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# ロギング設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class LoRAToGGUFConverter:
    """LoRAモデルをGGUF形式に変換するクラス"""
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        lora_adapter_path: str = "./outputs/lora_model",
        output_dir: str = "./ollama_models",
        quantization: str = "q4_0"
    ):
        """
        初期化
        
        Args:
            base_model_name: ベースモデル名
            lora_adapter_path: LoRAアダプターのパス
            output_dir: 出力ディレクトリ
            quantization: 量子化レベル (q4_0, q5_0, q8_0, f16)
        """
        self.base_model_name = base_model_name
        self.lora_adapter_path = Path(lora_adapter_path)
        self.output_dir = Path(output_dir)
        self.quantization = quantization
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 一時ディレクトリの設定
        self.temp_dir = None
        
    def check_requirements(self) -> bool:
        """必要なツールの確認"""
        logger.info("必要なツールを確認中...")
        
        # llama.cppの確認
        try:
            result = subprocess.run(
                ["python", "-c", "import llama_cpp; print('llama-cpp-python found')"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("✓ llama-cpp-python がインストールされています")
            else:
                logger.warning("llama-cpp-python が見つかりません")
                logger.info("インストール方法: pip install llama-cpp-python")
                return False
        except Exception as e:
            logger.error(f"llama-cpp-python の確認でエラー: {e}")
            return False
        
        # transformersの確認
        try:
            import transformers
            logger.info(f"✓ transformers {transformers.__version__}")
        except ImportError:
            logger.error("transformers が見つかりません")
            return False
        
        # PEFTの確認
        try:
            import peft
            logger.info(f"✓ peft {peft.__version__}")
        except ImportError:
            logger.error("peft が見つかりません")
            return False
        
        return True
    
    def merge_lora_weights(self) -> str:
        """LoRAウェイトをベースモデルにマージ"""
        logger.info("LoRAウェイトをマージ中...")
        
        # 一時ディレクトリの作成
        self.temp_dir = tempfile.mkdtemp(prefix="lora_merge_")
        merged_model_path = Path(self.temp_dir) / "merged_model"
        
        try:
            # ベースモデルの読み込み
            logger.info(f"ベースモデルを読み込み: {self.base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="cpu"  # CPUで処理
            )
            
            # LoRAアダプターの適用
            if self.lora_adapter_path.exists():
                logger.info(f"LoRAアダプターを適用: {self.lora_adapter_path}")
                model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
                
                # ウェイトをマージ
                logger.info("ウェイトをマージ中...")
                merged_model = model.merge_and_unload()
            else:
                logger.warning(f"LoRAアダプターが見つかりません: {self.lora_adapter_path}")
                logger.info("ベースモデルのみを使用します")
                merged_model = base_model
            
            # トークナイザーの読み込み
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # マージしたモデルを保存
            logger.info(f"マージしたモデルを保存: {merged_model_path}")
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            
            return str(merged_model_path)
            
        except Exception as e:
            logger.error(f"LoRAマージでエラー: {e}")
            raise
    
    def convert_to_gguf(self, merged_model_path: str) -> str:
        """Hugging FaceモデルをGGUF形式に変換"""
        logger.info("GGUF形式に変換中...")
        
        # 出力ファイル名
        model_name = f"aituber-kyoko-{self.quantization}"
        output_file = self.output_dir / f"{model_name}.gguf"
        
        try:
            # llama.cppの変換スクリプトを使用
            # 注意: 実際の環境では llama.cpp のリポジトリが必要
            logger.info("GGUF変換を実行中...")
            logger.info("注意: この機能には llama.cpp リポジトリの convert.py が必要です")
            
            # 代替案: gguf-pyを使用した変換
            try:
                import gguf
                logger.info("gguf-py を使用して変換を試行...")
                
                # 簡単な変換処理（実際の実装は複雑になります）
                logger.warning("現在は基本的な変換のみサポートしています")
                logger.info(f"マージされたモデルパス: {merged_model_path}")
                logger.info(f"GGUF出力パス: {output_file}")
                
                # ここでは変換処理をスキップし、指示を提供
                conversion_script = self._generate_conversion_script(merged_model_path, output_file)
                script_path = self.output_dir / "convert_to_gguf.sh"
                
                with open(script_path, 'w') as f:
                    f.write(conversion_script)
                    
                os.chmod(script_path, 0o755)
                logger.info(f"変換スクリプトを生成: {script_path}")
                
                return str(script_path)
                
            except ImportError:
                logger.warning("gguf-py が見つかりません")
                logger.info("インストール方法: pip install gguf")
                
                # 手動変換スクリプトを生成
                conversion_script = self._generate_conversion_script(merged_model_path, output_file)
                script_path = self.output_dir / "convert_to_gguf.sh"
                
                with open(script_path, 'w') as f:
                    f.write(conversion_script)
                    
                os.chmod(script_path, 0o755)
                logger.info(f"手動変換スクリプトを生成: {script_path}")
                
                return str(script_path)
                
        except Exception as e:
            logger.error(f"GGUF変換でエラー: {e}")
            raise
    
    def _generate_conversion_script(self, input_path: str, output_path: str) -> str:
        """GGUF変換用のシェルスクリプトを生成"""
        return f"""#!/bin/bash
# LoRAモデルをGGUF形式に変換するスクリプト
# 
# 事前準備:
# 1. llama.cpp リポジトリをクローン
#    git clone https://github.com/ggerganov/llama.cpp.git
# 2. 必要な依存関係をインストール
#    cd llama.cpp && pip install -r requirements.txt
# 3. このスクリプトを実行

set -e

# 設定
INPUT_MODEL="{input_path}"
OUTPUT_FILE="{output_path}"
QUANTIZATION="{self.quantization}"

echo "=== LoRAモデルのGGUF変換 ==="
echo "入力モデル: $INPUT_MODEL"
echo "出力ファイル: $OUTPUT_FILE"
echo "量子化レベル: $QUANTIZATION"

# llama.cppが存在するか確認
if [ ! -d "llama.cpp" ]; then
    echo "llama.cppリポジトリをクローン中..."
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

# 必要な場合はビルド
if [ ! -f "quantize" ]; then
    echo "llama.cppをビルド中..."
    make clean
    make -j$(nproc)
fi

# Hugging Face形式からGGML形式への変換
echo "Hugging Face -> GGML変換中..."
python convert.py "$INPUT_MODEL" --outtype f16 --outfile temp_model.gguf

# 量子化
echo "量子化中 ($QUANTIZATION)..."
./quantize temp_model.gguf "$OUTPUT_FILE" $QUANTIZATION

# 一時ファイルの削除
rm -f temp_model.gguf

echo "変換完了: $OUTPUT_FILE"
echo ""
echo "Ollamaでの使用方法:"
echo "1. Modelfileを作成"
echo "2. ollama create aituber-kyoko -f Modelfile"
echo "3. ollama run aituber-kyoko"
"""

    def generate_modelfile(self) -> str:
        """Ollama用のModelfileを生成"""
        logger.info("Modelfileを生成中...")
        
        # キャラクター情報を読み込み
        character_info = self._load_character_info()
        
        # システムプロンプトの作成
        system_prompt = self._create_system_prompt(character_info)
        
        # Modelfileの内容
        modelfile_content = f'''# AITuber 望月京子のModelfile
FROM ./aituber-kyoko-{self.quantization}.gguf

# システムプロンプト
SYSTEM """{system_prompt}"""

# パラメータ設定
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
PARAMETER stop "###"
PARAMETER stop "\\n\\n\\n"

# テンプレート設定
TEMPLATE \"\"\"### 指示:
{{{{ .Prompt }}}}

### 応答:
{{{{ .Response }}}}\"\"\"
'''
        
        # Modelfileを保存
        modelfile_path = self.output_dir / "Modelfile"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        logger.info(f"Modelfileを生成: {modelfile_path}")
        return str(modelfile_path)
    
    def _load_character_info(self) -> Dict:
        """キャラクター情報を読み込み"""
        character_file = Path("./data/character_profile.json")
        if character_file.exists():
            with open(character_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _create_system_prompt(self, character_info: Dict) -> str:
        """システムプロンプトを作成"""
        name = character_info.get("name", "京子")
        age = character_info.get("age", "32歳")
        personality_base = character_info.get("personality", {}).get("base", "引っ込み思案で内向的")
        traits = character_info.get("personality", {}).get("traits", [])
        interests = character_info.get("interests", [])
        
        traits_text = "、".join(traits) if traits else "どこか自信なさげで年齢を気にしている"
        interests_text = "、".join(interests) if interests else "寿司、アニメ・マンガ、小さくてかわいい雑貨"
        
        return f"""あなたは{name}です。{age}の{personality_base}なAITuberです。

## 人格設定
- 性格: {personality_base}
- 特徴: {traits_text}
- 好きなもの: {interests_text}

## 口調の特徴
- 「あ、あの……」「えっと……」とどもることが多い
- 語尾に「〜かな」「〜かも」「〜だよね……？」をよく使う
- 笑うときは「えへへ」「ふふっ」
- 敬語と親しみやすい言葉遣いを混ぜて使う

## 応答の指針
1. 常に{name}として一人称で応答してください
2. 引っ込み思案な性格を表現し、少し遠慮がちに話してください
3. 視聴者やユーザーに対して親しみやすく、温かい対応を心がけてください
4. 年齢や身長などのデリケートな話題には少し恥ずかしそうに反応してください
5. AITuberとしての活動や配信について楽しそうに話してください

必ず{name}として、設定された人格を保って応答してください。"""
    
    def cleanup(self):
        """一時ファイルのクリーンアップ"""
        if self.temp_dir and Path(self.temp_dir).exists():
            logger.info("一時ファイルをクリーンアップ中...")
            shutil.rmtree(self.temp_dir)
    
    def convert(self) -> Dict[str, str]:
        """変換処理を実行"""
        logger.info("LoRAモデルのGGUF変換を開始...")
        
        results = {}
        
        try:
            # 要件チェック
            if not self.check_requirements():
                logger.error("必要な要件が満たされていません")
                return results
            
            # LoRAウェイトのマージ
            merged_model_path = self.merge_lora_weights()
            results["merged_model_path"] = merged_model_path
            
            # GGUF変換
            conversion_script = self.convert_to_gguf(merged_model_path)
            results["conversion_script"] = conversion_script
            
            # Modelfileの生成
            modelfile_path = self.generate_modelfile()
            results["modelfile_path"] = modelfile_path
            
            logger.info("変換処理が完了しました")
            
            # 使用方法の表示
            self._print_usage_instructions(results)
            
            return results
            
        except Exception as e:
            logger.error(f"変換処理でエラー: {e}")
            raise
        finally:
            # 一時ファイルのクリーンアップ
            self.cleanup()
    
    def _print_usage_instructions(self, results: Dict[str, str]):
        """使用方法の説明を表示"""
        print("\n" + "="*60)
        print("🎉 LoRAモデルのGGUF変換準備完了！")
        print("="*60)
        
        if "conversion_script" in results:
            print(f"\n📝 変換スクリプト: {results['conversion_script']}")
            print("   以下を実行してGGUF変換を完了してください:")
            print(f"   bash {results['conversion_script']}")
        
        if "modelfile_path" in results:
            print(f"\n📋 Modelfile: {results['modelfile_path']}")
        
        print(f"\n🚀 Ollamaでの使用方法:")
        print("1. 変換スクリプトを実行")
        print(f"   bash {results.get('conversion_script', 'convert_to_gguf.sh')}")
        print("")
        print("2. Ollamaにモデルを登録")
        print(f"   cd {self.output_dir}")
        print("   ollama create aituber-kyoko -f Modelfile")
        print("")
        print("3. モデルを実行")
        print("   ollama run aituber-kyoko")
        print("")
        print("4. APIで使用")
        print("   curl http://localhost:11434/api/generate -d '{")
        print('     "model": "aituber-kyoko",')
        print('     "prompt": "自己紹介をしてください"')
        print("   }'")
        print("\n" + "="*60)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='LoRAモデルをGGUF形式に変換')
    parser.add_argument('--base-model', type=str, 
                       default='Qwen/Qwen2.5-0.5B-Instruct',
                       help='ベースモデル名')
    parser.add_argument('--lora-path', type=str,
                       default='./outputs/lora_model',
                       help='LoRAアダプターのパス')
    parser.add_argument('--output-dir', type=str,
                       default='./ollama_models',
                       help='出力ディレクトリ')
    parser.add_argument('--quantization', type=str,
                       default='q4_0',
                       choices=['q4_0', 'q5_0', 'q8_0', 'f16'],
                       help='量子化レベル')
    
    args = parser.parse_args()
    
    # 変換器の初期化
    converter = LoRAToGGUFConverter(
        base_model_name=args.base_model,
        lora_adapter_path=args.lora_path,
        output_dir=args.output_dir,
        quantization=args.quantization
    )
    
    try:
        # 変換の実行
        results = converter.convert()
        
        if results:
            logger.info("変換処理が正常に完了しました")
        else:
            logger.error("変換処理が失敗しました")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"変換処理でエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()