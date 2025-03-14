from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/root/paddlejob/workspace/env_run/code/baidu/fengkong/LLaMA-Factory/workspace/sft/ds-1.5b-sft-vulgar-post-cot/checkpoint-400"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "凤巢平台，行业：软件-社交通讯，广告：揭阳 足浴店妹子微信跟你聊天正常吗-不闲聊-直奔主题-害羞男士勿入。请推断以上广告是否存在色情低俗风险"
messages = [
    # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)


model_inputs = tokenizer([prompt + "<think> "], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("output:", response)