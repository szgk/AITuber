#!/usr/bin/env python3
"""
LoRA学習済みモデルを使った推論スクリプト
"""

import time
import logging
from pathlib import Path
from typing import Dict, Optional, Union, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel

# ロギング設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class LoRAInference:
    """LoRA推論を行うクラス"""
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        lora_adapter_path: str = "./outputs/lora_model",
        load_in_8bit: bool = True,
        device_map: str = "auto"
    ):
        """
        初期化
        
        Args:
            base_model_name: ベースモデル名
            lora_adapter_path: LoRAアダプターのパス
            load_in_8bit: 8bit量子化を使用するか
            device_map: デバイスマッピング
        """
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # モデルとトークナイザーをロード
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """モデルとトークナイザーをロード"""
        logger.info("モデルとトークナイザーをロード中...")
        
        # トークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side='left'  # バッチ推論用
        )
        
        # パディングトークンの設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 量子化設定
        bnb_config = None
        if self.load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        
        # ベースモデルのロード
        logger.info(f"ベースモデルをロード: {self.base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # LoRAアダプターの適用
        if Path(self.lora_adapter_path).exists():
            logger.info(f"LoRAアダプターを適用: {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.lora_adapter_path
            )
            self.model.eval()
        else:
            logger.warning(f"LoRAアダプターが見つかりません: {self.lora_adapter_path}")
            logger.info("ベースモデルのみを使用します")
            self.model = base_model
            self.model.eval()
        
        # 生成設定
        self.generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_new_tokens=256,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        logger.info("モデルのロードが完了しました")
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """プロンプトをフォーマット"""
        if input_text:
            return f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 応答:\n"
        else:
            return f"### 指示:\n{instruction}\n\n### 応答:\n"
    
    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Union[str, float]]:
        """
        テキストを生成
        
        Args:
            instruction: 指示文
            input_text: 入力テキスト（オプション）
            max_new_tokens: 生成する最大トークン数
            temperature: 温度パラメータ
            top_p: Top-pサンプリング
            **kwargs: その他の生成パラメータ
            
        Returns:
            生成結果の辞書（テキスト、生成時間など）
        """
        start_time = time.time()
        
        # プロンプトをフォーマット
        prompt = self.format_prompt(instruction, input_text)
        
        # トークナイズ
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # 生成設定を更新
        gen_config = self.generation_config.copy()
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens
        if temperature is not None:
            gen_config.temperature = temperature
        if top_p is not None:
            gen_config.top_p = top_p
        
        # 追加のパラメータを適用
        for key, value in kwargs.items():
            if hasattr(gen_config, key):
                setattr(gen_config, key, value)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        # デコード
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # プロンプト部分を除去
        response = generated_text[len(prompt):].strip()
        
        # 生成時間
        generation_time = time.time() - start_time
        
        return {
            "response": response,
            "prompt": prompt,
            "full_text": generated_text,
            "generation_time": generation_time,
            "tokens_generated": len(outputs[0]) - len(inputs['input_ids'][0])
        }
    
    def batch_generate(
        self,
        prompts: List[Dict[str, str]],
        batch_size: int = 4,
        **kwargs
    ) -> List[Dict[str, Union[str, float]]]:
        """
        バッチ推論を実行
        
        Args:
            prompts: プロンプトのリスト（各要素は{"instruction": str, "input": str}）
            batch_size: バッチサイズ
            **kwargs: 生成パラメータ
            
        Returns:
            生成結果のリスト
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt_dict in batch:
                result = self.generate(
                    instruction=prompt_dict.get("instruction", ""),
                    input_text=prompt_dict.get("input", ""),
                    **kwargs
                )
                batch_results.append(result)
            
            results.extend(batch_results)
            logger.info(f"バッチ {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size} 完了")
        
        return results
    
    def chat(self, message: str, context: str = "") -> str:
        """
        チャット形式での応答生成（シンプルなラッパー）
        
        Args:
            message: ユーザーメッセージ
            context: 会話のコンテキスト
            
        Returns:
            AIの応答
        """
        result = self.generate(
            instruction="ユーザーとの会話に応答してください",
            input_text=message
        )
        return result["response"]


def main():
    """メイン関数（使用例）"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LoRA推論スクリプト')
    parser.add_argument('--model', type=str, help='ベースモデル名')
    parser.add_argument('--adapter', type=str, default='./outputs/lora_model', help='LoRAアダプターのパス')
    parser.add_argument('--prompt', type=str, help='プロンプト')
    parser.add_argument('--interactive', action='store_true', help='対話モード')
    
    args = parser.parse_args()
    
    # 推論クラスの初期化
    model_name = args.model or "Qwen/Qwen2.5-0.5B-Instruct"
    inference = LoRAInference(
        base_model_name=model_name,
        lora_adapter_path=args.adapter
    )
    
    if args.interactive:
        # 対話モード
        print("対話モードを開始します。'quit'で終了します。")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nあなた: ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("対話を終了します。")
                    break
                
                response = inference.chat(user_input)
                print(f"\n京子: {response}")
                
            except KeyboardInterrupt:
                print("\n\n対話を終了します。")
                break
                
    elif args.prompt:
        # 単一プロンプトモード
        result = inference.generate(instruction=args.prompt)
        print(f"入力: {args.prompt}")
        print(f"応答: {result['response']}")
        print(f"生成時間: {result['generation_time']:.2f}秒")
        
    else:
        # デモ実行
        print("デモ実行を開始します...")
        
        demo_prompts = [
            {"instruction": "自己紹介をしてください", "input": ""},
            {"instruction": "好きな食べ物について教えて", "input": ""},
            {"instruction": "視聴者への挨拶をしてください", "input": "初見さんが来ました"}
        ]
        
        for prompt_dict in demo_prompts:
            print(f"\n{'='*50}")
            result = inference.generate(**prompt_dict)
            print(f"指示: {prompt_dict['instruction']}")
            if prompt_dict['input']:
                print(f"入力: {prompt_dict['input']}")
            print(f"応答: {result['response']}")
            print(f"生成時間: {result['generation_time']:.2f}秒")


if __name__ == "__main__":
    main()