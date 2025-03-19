# 环境安装（需要提前安装）
# pip install transformers datasets trl accelerate torch peft
import json
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
import torch

# 参数配置
model_name = "/root/paddlejob/workspace/env_run/code/baidu/fengkong/nlp-model/llm/model/DeepSeek-R1-Distill-Qwen-1.5B"  # 根据实际选择模型大小
dataset_name = "../data/sharegpt_eval_post_cot.json"  # 替换为你的数据集
output_dir = "./qwen2.5-sft-checkpoints"
max_seq_length = 512  # 根据GPU显存调整

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 设置padding token

# 量化配置（可选，减少显存消耗）
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #quantization_config=bnb_config,  # 注释掉此行禁用量化
    device_map="auto",
    trust_remote_code=True
)

# 加载数据集（示例使用json格式）
# dataset = load_dataset("json", data_files={"train": "path_to_your_data.json"})
with open(dataset_name, 'r', encoding='utf-8') as f:
    data = json.load(f)
data_list = []
for conversation in data:
    prompt = response = ""
    for turn in conversation['conversations']:
        if turn['from'] == 'human':
            prompt = turn['value']
        if turn['from'] == 'gpt':
            response = turn['value']
    data_list.append([prompt, response])
df = pd.DataFrame(data[:32], columns=["prompt", "response"])
dataset = Dataset.from_pandas(df)


# 数据预处理函数
def format_instruction(sample):
    # 根据你的数据集结构调整
    # return f"""<|im_start|>user{sample['prompt']}<|im_end|><|im_start|>assistant{sample['response']}<|im_end|>"""
    formatted_text = f"<｜begin▁of▁sentence｜><｜User｜>{sample['prompt']}<｜Assistant｜>{sample['response']}<｜end▁of▁sentence｜>"
    return formatted_text

# 训练参数配置
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,    # 根据显存调整
    gradient_accumulation_steps=2,    # 根据显存调整
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=True,                        # 如果支持混合精度
    remove_unused_columns=False,
    optim="adamw_torch",
    report_to="tensorboard",
    max_steps=-1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)

# 初始化训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    formatting_func=format_instruction,  # 应用数据格式模板
    dataset_text_field=None            # 如果数据集已有格式化文本字段
)

# 开始训练
trainer.train()

# 保存微调后的模型
trainer.save_model("./qwen2.5-sft-final")

# 使用示例
input_text = f"<｜begin▁of▁sentence｜><｜User｜>凤巢平台，行业：软件-社交通讯，广告：揭阳 足浴店妹子微信跟你聊天正常吗-不闲聊-直奔主题-害羞男士勿入。请推断以上广告是否存在色情低俗风险<｜Assistant｜>"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))