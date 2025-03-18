import json
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def model_infer_batch(model, tokenizer, batch_messages):
    """处理单个批次并返回结果"""
    # 生成批量prompt
    batch_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) for messages in batch_messages
    ]

    # 批量编码
    model_inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    ).to(model.device)

    # 获取实际输入长度
    input_lengths = model_inputs.attention_mask.sum(dim=1)

    # 批量生成
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id  # 添加结束符号作为解码终止标志
    )

    # 解码结果
    results = []
    for i, seq in enumerate(generated_ids):
        output_length = max(input_lengths)
        response = tokenizer.decode(
            seq[output_length:], 
            skip_special_tokens=True
        )
        # 构造包含输入输出的完整记录
        record = {
            "input": batch_texts[i],
            "output": response.strip()
        }
        results.append(record)
    
    return results

def data_load(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    batch_messages = []
    for conversation in data:
        for turn in conversation['conversations']:
            if turn['from'] == 'human':
                message = turn['value']
                batch_messages.append([{"role": "user", "content": message}])
                break
    return batch_messages

def main():
    input_file = "../data/sharegpt_eval_post_cot.json"     # 每行一个对话样本的JSON文件
    output_file = "./data/result_sharegpt_eval_post_cot_test_v2.json"   # 输出结果文件
    batch_size = 16                # 批处理大小
    model_name = "/root/paddlejob/workspace/env_run/code/baidu/fengkong/LLaMA-Factory/workspace/sft/ds-1.5b-sft-vulgar-post-cot/checkpoint-400"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"  # 左侧padding
    )

    # all_messages = data_load(input_file)
    # debug
    all_messages = data_load(input_file)[:16]
    print("eos_token_id: ", tokenizer.eos_token_id)
    print(all_messages)

    # 创建进度条
    total = len(all_messages)
    # progress_bar = tqdm(total=total, desc="Processing")
    progress_bar = tqdm(total=total, desc="Processing", mininterval=0.1, maxinterval=1)

    st = time.time()
    # 批量处理并保存结果
    with open(output_file, "w") as out_f:
        for i in range(0, total, batch_size):
            batch = all_messages[i: i + batch_size]
            
            try:
                batch_results = model_infer_batch(model, tokenizer, batch)
                # 写入结果
                for record in batch_results:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # 更新进度
                progress_bar.update(len(batch))
                
            except Exception as e:
                print(f"Error processing batch {i // batch_size}: {str(e)}")
                continue

    progress_bar.close()
    print(f"\nProcessing completed! Results saved to {output_file}, used time:{time.time()-st}")

if __name__ == "__main__":
    main()
