import json
import time
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

def data_load(input_file, tokenizer):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    batch_messages = []
    for conversation in data:
        messages = []
        for turn in conversation['conversations']:
            if turn['from'] == 'human':
                messages.append({"role": "user", "content": turn['value']})
                break
        if messages:
            # 构建vLLM需要的prompt格式
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_messages.append({
                "prompt": prompt,
                "original": messages  # 保留原始信息
            })
    return batch_messages

def main():
    input_file = "./data/zeus_benchmark.json"
    output_file = "./data/result_benchmark_vllm_test.json"
    batch_size = 16
    model_name = "/root/paddlejob/workspace/env_run/code/baidu/fengkong/LLaMA-Factory/workspace/sft/ds-1.5b-sft-vulgar-post-cot/checkpoint-400"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True
    )
    # 初始化vLLM
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        tensor_parallel_size=1,  # GPU数量
        dtype="auto",
        max_model_len=512,      # 最大上下文长度
        gpu_memory_utilization=0.9  # GPU内存利用率
    )

    # 配置生成参数
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.1,
        top_p=0.9,
        stop_token_ids=[151643]  # 根据实际tokenizer设置
    )

    # 加载数据
    all_data = data_load(input_file, tokenizer)[:16]
    prompts = [item["prompt"] for item in all_data]
    originals = [item["original"] for item in all_data]

    print("all_data:", all_data)
    print("prompts:", prompts)


    # 批量推理
    st = time.time()
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing"):
        batch_prompts = prompts[i : i+batch_size]
        print(batch_prompts)

        outputs = llm.generate(
            batch_prompts,
            sampling_params,
            use_tqdm=False  # 禁用内部进度条
        )

        # 处理结果
        for output, original in zip(outputs, originals[i:i+batch_size]):
            results.append({
                "input": original,
                "output": output.outputs[0].text.strip()
            })

    # 保存结果
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # print(f"Processing completed! Throughput: {len(prompts)/llm.llm_engine.stats.total_time:.2f} prompts/s")
    print(f"used time:{time.time()-st} seconds")

if __name__ == "__main__":
    main()