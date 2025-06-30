#!/usr/bin/env python3
"""
LoRA学習スクリプト
Qwen3-4BモデルをベースにLoRAでファインチューニングを行う
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
from accelerate import Accelerator

# ロギング設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class LoRATrainer:
    """LoRA学習を管理するクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス（指定しない場合はデフォルト設定を使用）
        """
        self.config = self.load_config(config_path)
        self.accelerator = Accelerator()
        
    def load_config(self, config_path: Optional[str]) -> Dict:
        """設定を読み込む"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"設定ファイルを読み込みました: {config_path}")
        else:
            # デフォルト設定
            config = {
                'model': {
                    'name': 'Qwen/Qwen2.5-0.5B-Instruct',  # テスト用の小さいモデル
                    'load_in_8bit': True,
                    'device_map': 'auto'
                },
                'lora': {
                    'r': 16,
                    'lora_alpha': 32,
                    'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                    'lora_dropout': 0.1,
                    'bias': 'none',
                    'task_type': 'CAUSAL_LM'
                },
                'training': {
                    'output_dir': './outputs/lora_model',
                    'num_train_epochs': 3,
                    'per_device_train_batch_size': 4,
                    'per_device_eval_batch_size': 4,
                    'gradient_accumulation_steps': 1,
                    'learning_rate': 2e-4,
                    'warmup_steps': 100,
                    'logging_steps': 10,
                    'save_steps': 100,
                    'eval_steps': 100,
                    'save_total_limit': 3,
                    'fp16': True,
                    'optim': 'adamw_torch',
                    'gradient_checkpointing': True,
                    'report_to': 'none'
                },
                'data': {
                    'train_file': './data/training/train.jsonl',
                    'val_file': './data/training/validation.jsonl',
                    'max_length': 512
                }
            }
            logger.info("デフォルト設定を使用します")
        
        return config
    
    def load_data(self, file_path: str) -> List[Dict]:
        """JSONLファイルからデータを読み込む"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        logger.info(f"{file_path}から{len(data)}件のデータを読み込みました")
        return data
    
    def format_instruction(self, example: Dict) -> str:
        """データを学習用フォーマットに変換"""
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        if input_text:
            text = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 応答:\n{output}"
        else:
            text = f"### 指示:\n{instruction}\n\n### 応答:\n{output}"
        
        return text
    
    def prepare_datasets(self, tokenizer):
        """データセットを準備"""
        # データを読み込み
        train_data = self.load_data(self.config['data']['train_file'])
        val_data = self.load_data(self.config['data']['val_file'])
        
        # フォーマット変換
        train_texts = [self.format_instruction(example) for example in train_data]
        val_texts = [self.format_instruction(example) for example in val_data]
        
        # トークナイズ
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.config['data']['max_length']
            )
        
        # Dataset作成
        train_dataset = Dataset.from_dict({'text': train_texts})
        val_dataset = Dataset.from_dict({'text': val_texts})
        
        # トークナイズ
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        logger.info(f"訓練データ: {len(train_dataset)}件")
        logger.info(f"検証データ: {len(val_dataset)}件")
        
        return train_dataset, val_dataset
    
    def setup_model_and_tokenizer(self):
        """モデルとトークナイザーを設定"""
        model_name = self.config['model']['name']
        
        # トークナイザーの読み込み
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='right'
        )
        
        # パディングトークンの設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 量子化設定
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=self.config['model']['load_in_8bit'],
            bnb_8bit_compute_dtype=torch.float16
        )
        
        # モデルの読み込み
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=self.config['model']['device_map'],
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 勾配チェックポイントの有効化
        if self.config['training']['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
        
        # 8bit学習の準備
        model = prepare_model_for_kbit_training(model)
        
        # LoRA設定
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        # LoRAモデルの作成
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model, tokenizer
    
    def train(self):
        """学習を実行"""
        logger.info("学習を開始します...")
        
        # モデルとトークナイザーの準備
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # データセットの準備
        train_dataset, val_dataset = self.prepare_datasets(tokenizer)
        
        # データコレーターの準備
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # 学習引数の設定（型変換を含む）
        training_args_dict = {
            'output_dir': self.config['training']['output_dir'],
            'num_train_epochs': int(self.config['training']['num_train_epochs']),
            'per_device_train_batch_size': int(self.config['training']['per_device_train_batch_size']),
            'per_device_eval_batch_size': int(self.config['training']['per_device_eval_batch_size']),
            'gradient_accumulation_steps': int(self.config['training']['gradient_accumulation_steps']),
            'learning_rate': float(self.config['training']['learning_rate']),
            'warmup_steps': int(self.config['training']['warmup_steps']),
            'logging_steps': int(self.config['training']['logging_steps']),
            'save_steps': int(self.config['training']['save_steps']),
            'eval_steps': int(self.config['training']['eval_steps']),
            'save_total_limit': int(self.config['training']['save_total_limit']),
            'fp16': bool(self.config['training']['fp16']),
            'optim': str(self.config['training']['optim']),
            'load_best_model_at_end': True,
            'report_to': str(self.config['training']['report_to']),
            'remove_unused_columns': False
        }
        
        # transformersのバージョンに応じて互換性のあるパラメータを設定
        try:
            import transformers
            version = transformers.__version__
            logger.info(f"Transformers version: {version}")
            
            # バージョン4.19以降でevaluation_strategyがサポート
            if hasattr(TrainingArguments, 'evaluation_strategy'):
                training_args_dict['evaluation_strategy'] = "steps"
            elif hasattr(TrainingArguments, 'eval_strategy'):
                training_args_dict['eval_strategy'] = "steps"
            else:
                # 古いバージョンでは評価戦略を無効化
                logger.warning("evaluation_strategy not supported, disabling evaluation")
                training_args_dict.pop('eval_steps', None)
                training_args_dict['load_best_model_at_end'] = False
                
        except Exception as e:
            logger.warning(f"バージョンチェックでエラー: {e}")
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Trainerの作成（処理クラス名を更新）
        trainer_kwargs = {
            'model': model,
            'args': training_args,
            'train_dataset': train_dataset,
            'eval_dataset': val_dataset,
            'data_collator': data_collator
        }
        
        # transformersバージョンに応じてtokenizerまたはprocessing_classを設定
        try:
            import inspect
            trainer_sig = inspect.signature(Trainer.__init__)
            if 'processing_class' in trainer_sig.parameters:
                trainer_kwargs['processing_class'] = tokenizer
            else:
                trainer_kwargs['tokenizer'] = tokenizer
        except:
            # フォールバック: 古い形式を使用
            trainer_kwargs['tokenizer'] = tokenizer
        
        trainer = Trainer(**trainer_kwargs)
        
        # 学習の実行
        trainer.train()
        
        # モデルの保存
        logger.info("モデルを保存しています...")
        trainer.save_model()
        tokenizer.save_pretrained(self.config['training']['output_dir'])
        
        logger.info(f"学習が完了しました。モデルは{self.config['training']['output_dir']}に保存されました。")
        
        return trainer


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LoRA学習スクリプト')
    parser.add_argument(
        '--config',
        type=str,
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='使用するモデル名（設定ファイルの値を上書き）'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='学習エポック数（設定ファイルの値を上書き）'
    )
    
    args = parser.parse_args()
    
    # トレーナーの初期化
    trainer = LoRATrainer(args.config)
    
    # コマンドライン引数で設定を上書き
    if args.model:
        trainer.config['model']['name'] = args.model
        logger.info(f"モデルを{args.model}に変更しました")
    
    if args.epochs:
        trainer.config['training']['num_train_epochs'] = args.epochs
        logger.info(f"エポック数を{args.epochs}に変更しました")
    
    # 学習の実行
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"学習中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()