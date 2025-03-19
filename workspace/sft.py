# -*- coding: utf-8 -*-
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

# 参数配置
MODEL_NAME = "/root/paddlejob/workspace/env_run/code/baidu/fengkong/LLaMA-Factory/workspace/sft/ds-1.5b-sft-vulgar-post-cot/checkpoint-400"  # 支持0.5B到72B不同规模模型
TEMPLATE = f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜>"  # 官方推荐模板[6](@ref)


# 加载预训练模型和分词器
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False  # 确保兼容特殊控制符
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # 提升长文本效率[4](@ref)
)

# 数据预处理函数
def preprocess_function(examples):
    texts = []
    for ins, inp, out in zip(examples["instruction"], 
                           examples["input"],
                           examples["output"]):
        # 结构化数据转换[4](@ref)
        prompt = TEMPLATE.format(
            instruction=ins,
            input=f"\n相关数据：{inp}" if inp else "",
            output=out
        )
        texts.append(prompt)
    
    # 动态分块（支持1M上下文）
    tokenized = tokenizer(
        texts,
        max_length=32768,  # 基础上下文长度
        truncation=True,
        padding=False,
        add_special_tokens=False
    )
    return tokenized

# 加载数据集（示例格式）
dataset = load_dataset("json", data_files="sft_data.json")["train"]
dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["instruction", "input", "output"]
)

# 数据拆分（8:2比例）
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 训练参数配置（适配Qwen2.5特性[4](@ref)）
args = TrainingArguments(
    output_dir="./qwen2.5-sft",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=3,
    logging_steps=50,
    fp16_full_eval=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    optim="adamw_torch_fused",
    max_grad_norm=0.5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="none",
    gradient_checkpointing=True  # 节省显存[6](@ref)
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

# 开始微调
trainer.train()

# 保存适配器权重（支持增量保存）
model.save_pretrained("./qwen2.5-sft/final_model")
tokenizer.save_pretrained("./qwen2.5-sft/final_model")