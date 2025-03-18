import json
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def pr(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    print("base", "p:", precision, "r:", recall, "f1:", f1)

def all_pr(y_true, y_pred):
    y_true = [1 if x == 2 else x for x in y_true]
    y_pred = [1 if x == 2 else x for x in y_pred]

    precision_micro = precision_score(y_true, y_pred, average=None)
    recall_micro = recall_score(y_true, y_pred, average=None)
    f1_micro = f1_score(y_true, y_pred, average=None)
    print("merge p:", precision_micro, "r:", recall_micro, "f1:", f1_micro)

# 读取 JSON 文件
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 解析对话内容
def parse_conversations(data):
    for conversation in data:
        print(f"Conversation ID: {conversation['id']}")
        for turn in conversation['conversations']:
            speaker = turn['from']
            message = turn['value']
            print(f"{speaker}: {message}")
        print("-" * 40)  # 分隔线

# 主函数
def main():
    file_path = 'sharegpt_data.json'  # 替换为你的 JSON 文件路径
    data = load_json_file(file_path)
    parse_conversations(data)

def label_process(label_list):
    result_list = []
    for i in label_list:
        result = -1
        if "底线" in i:
            result = 2
        elif "擦边" in i:
            result = 1
        elif "无" in i:
            result = 0
        result_list.append(result)
    return result_list

# 结果1
file_path = "../data/sharegpt_eval_post_cot.json"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
label_list = []
for conversation in data:
    for turn in conversation['conversations']:
        if turn['from'] == 'gpt':
            label_list.append(turn['value'].split("</think>")[1].strip())
result_list = []
for i in open("./data/result_sharegpt_eval_post_cot.json"):
    case = json.loads(i.strip())
    result = ""
    try:
        result = case['output'].split("</think>\n\n")[1]
    except:
        pass
    result_list.append(result)

grownd_ture = label_process(label_list)
pred = label_process(result_list)

data = pd.DataFrame()
data["grownd_ture"] = grownd_ture
data["pred"] = pred

print("error num: ", (data["pred"] == -1).sum())
data = data[data["pred"] != -1]
pr(data["grownd_ture"], data["pred"])
all_pr(data["grownd_ture"], data["pred"])


# 结果2
file_path = "../data/sharegpt_eval_post_cot.json"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
label_list = []
for conversation in data:
    for turn in conversation['conversations']:
        if turn['from'] == 'gpt':
            label_list.append(turn['value'].split("</think>")[1].strip())
result_list = []
with open("./data/result_sharegpt_eval_post_cot_vllm.json", 'r', encoding='utf-8') as f:
    data_pred = json.load(f)
    for case in data_pred:
        result = ""
        try:
            result = case['output'].split("</think>\n\n")[1]
        except:
            pass
        result_list.append(result)

grownd_ture = label_process(label_list)
pred = label_process(result_list)

data = pd.DataFrame()
data["grownd_ture"] = grownd_ture
data["pred"] = pred

print("error num: ", (data["pred"] == -1).sum())
data = data[data["pred"] != -1]
pr(data["grownd_ture"], data["pred"])
all_pr(data["grownd_ture"], data["pred"])

file_path = "./data/zeus_benchmark.txt"
data = pd.read_csv(file_path, sep="\t")
data["grownd_ture"] = data["tgt"].apply(lambda x: int(x.split("[SEP]")[0]))

result_list = []
with open("./data/result_benchmark_vllm.json", 'r', encoding='utf-8') as f:
    data_pred = json.load(f)
    for case in data_pred:
        result = ""
        try:
            result = case['output'].split("</think>")[1].strip()
        except:
            pass
        result_list.append(result)

pred = label_process(result_list)
data["pred"] = pred

print("error num: ", (data["pred"] == -1).sum())
data = data[data["pred"] != -1]
pr(data["grownd_ture"], data["pred"])
all_pr(data["grownd_ture"], data["pred"])








