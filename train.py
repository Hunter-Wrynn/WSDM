# train.py

import os
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase, 
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score

# 配置类
@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "unsloth/gemma-2-9b-it-bnb-4bit"  # 4-bit quantized gemma-2-9b-instruct
    max_length: int = 1024
    n_splits: int = 8
    fold_idx: int = 0
    optim_type: str = "adamw_8bit"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2  # global batch size is 8 
    per_device_eval_batch_size: int = 8
    n_epochs: int = 1
    freeze_layers: int = 16  # there're 42 layers in total, we don't add adapters to the first 16 layers
    lr: float = 2e-4
    warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: float = 32  # lora_r * 2
    lora_dropout: float = 0.05
    lora_bias: str = "none"

# 自定义分词器类
class CustomTokenizer:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch: dict) -> dict:
        prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
        response_a = ["\n\n<response_a>: " + self.process_text(t) for t in batch["response_a"]]
        response_b = ["\n\n<response_b>: " + self.process_text(t) for t in batch["response_b"]]
        texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        labels = []
        for a_win, b_win in zip(batch["winner_model_a"], batch["winner_model_b"]):
            if a_win:
                label = 0
            elif b_win:
                label = 1
            else:
                label = 2
            labels.append(label)
        return {**tokenized, "labels": labels}
        
    @staticmethod
    def process_text(text: str) -> str:
        try:
            # 假设输入的 text 是一个字符串形式的列表
            return " ".join(eval(text, {"null": ""}))
        except Exception as e:
            print(f"Error processing text: {text}, error: {e}")
            return ""

# 评估指标计算函数
def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}

def main():
    # 初始化配置
    config = Config()
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        report_to="none",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=200,
        optim=config.optim_type,
        fp16=True,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
    )
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        # 仅目标自注意力层
        target_modules=["q_proj", "k_proj", "v_proj"],
        layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.SEQ_CLS,
    )
    
    # 初始化分词器
    tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
    tokenizer.add_eos_token = True  # 在末尾添加 <eos>
    tokenizer.padding_side = "right"
    
    # 加载基础模型
    model = Gemma2ForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=3,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    
    # 准备模型进行 k-bit 训练并应用 LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 可选：打印可训练参数
    
    # 加载数据集
    dataset_path = "../dataset/train.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件未找到: {dataset_path}")
    
    ds = Dataset.from_csv(dataset_path)
    ds = ds.select(list(range(100)))  # 仅使用前 100 条数据用于演示
    
    # 应用自定义分词器
    encode = CustomTokenizer(tokenizer, max_length=config.max_length)
    ds = ds.map(encode, batched=True)
    
    # 创建 K-Fold 划分
    folds = [
        (
            [i for i in range(len(ds)) if i % config.n_splits != fold_idx],
            [i for i in range(len(ds)) if i % config.n_splits == fold_idx]
        ) 
        for fold_idx in range(config.n_splits)
    ]
    
    train_idx, eval_idx = folds[config.fold_idx]
    
    # 初始化 Trainer
    trainer = Trainer(
        args=training_args, 
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds.select(train_idx),
        eval_dataset=ds.select(eval_idx),
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    # 训练模型
    print("开始训练...")
    trainer.train()
    
    # 评估模型在验证集上的性能
    print("开始评估...")
    eval_results = trainer.evaluate()
    
    # 打印评估结果
    print(f"Evaluation Results: {eval_results}")
    print(f"Accuracy: {eval_results.get('eval_acc', 'N/A')}")
    print(f"Log Loss: {eval_results.get('eval_log_loss', 'N/A')}")

if __name__ == "__main__":
    main()
