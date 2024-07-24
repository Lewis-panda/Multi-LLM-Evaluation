from glob import glob
import json
import re
from tqdm import tqdm
import os
from langchain_community.llms import Ollama

# 設置模型和對應的主機
model = "deepseek-v2:236b"
model_name = "deepseek"
host = "http://YourHost"

# 初始化 prompt 集合
prompt_sets = []
prompt_map = {}
with open("prompts/prompts_Eng.json", "r", encoding="utf-8") as r:
    prompts = json.load(r)
    for k, v in prompts.items():
        prompt_sets.append(k)
        prompt_map[k] = v

# 初始化模型
def init_model(model, host):
    generation_params = {
        "num_predict": 2048,
        "top_k": 25,
        "top_p": 0.6,
        "repeat_penalty": 1.2,
        "temperature": 0.65,
    }
    try:
        llm = Ollama(
            model=model,
            base_url=host,
            keep_alive=True,
            **generation_params
        )
        return llm
    except Exception as e:
        print(f"Failed to initialize model {model} on {host}: {e}")
        return None

def run_model(llm, prompt, content):
    content = re.sub(r"idx:\s\d+,\s", "", content)
    prompt = prompt.replace("<INSERT_EXTRACT>", content)
#    print(prompt)
    if len(prompt) == 0:
        raise ValueError("Prompt is empty.")

    if len(content) == 0:
        raise ValueError("Content is empty.")

    try:
        result = llm.invoke(prompt)
        return result
    except Exception as e:
        print(f"Error running model: {e}")
        return None

# 初始化模型
llm = init_model(model, host)
# 檢查模型是否初始化成功
if llm is None:
    raise RuntimeError(f"Failed to initialize model {model} on {host}")

# 初始化數據集集合
datasets = []
dataset_map = {}

# 遍歷所有分類文件夾並處理其中的 first50.jsonl 文件
for category_folder in glob('classify_data/*'):
    category_name = os.path.basename(category_folder)
    for ds_path in glob(f'{category_folder}/first50.jsonl', recursive=True):
        name = ds_path.split("/")[-1].split(".")[0]  # 獲取文件名，如 'first50'
        datasets.append(name)
        # 預讀模式
        samples = []
        with open(ds_path, "r", encoding="utf-8") as r:
            for l in tqdm(r, desc=f"Pre-loading the dataset in {category_name}!"):  # l: line -> every line in r
                sample = json.loads(l.strip())
                samples.append(sample["text"])
            dataset_map[f"{category_name}/{name}"] = samples

# 定義進度文件路徑
progress_file_dir = "records"
progress_file = f"{progress_file_dir}/{model_name}_ds_eng.pos.json"

# 確保進度文件目錄存在
os.makedirs(progress_file_dir, exist_ok=True)

# 如果進度文件不存在，則創建一個空字典並保存
if not os.path.exists(progress_file):
    ds_pos = {}
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(ds_pos, f, ensure_ascii=False, indent=4)
else:
    with open(progress_file, "r", encoding="utf-8") as r:
        ds_pos = json.load(r)

for k in dataset_map.keys():
    if k not in ds_pos:
        ds_pos[k] = 0

# 保存進度的函數
def save_progress(ds_pos, output_path=progress_file):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ds_pos, f, ensure_ascii=False, indent=4)

# 初始化結果列表
all_results = []

# 遍歷每個樣本
for category_name, content_list in dataset_map.items():
    # 選擇數據集和提示
    prompt = prompt_map.get(category_name.split('/')[0], "")
    if not prompt:
        print(f"No prompt found for category {category_name}")
        continue

    output_dir = f"RewriteResults/{model_name}/{category_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/first50_eng.json"
    
    # 如果文件已存在，讀取已有結果
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = []

    for idx, content in enumerate(tqdm(content_list, desc=f"({model_name})Processing samples in {category_name}")):
        if idx < ds_pos.get(f"{category_name}/first50", 0):
            continue
        result = run_model(llm, prompt, content)
        if result is not None:
            all_results.append(result)
            ds_pos[f"{category_name}/first50"] = idx + 1
            
            # 將結果保存到文件
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            # 保存進度
            save_progress(ds_pos)

    print(f"Results saved to {output_path}")

print("所有步驟完成")
