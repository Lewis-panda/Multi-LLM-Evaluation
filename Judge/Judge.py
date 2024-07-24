from litellm import completion, embedding
from openai import AzureOpenAI
import json
import os
from tqdm import tqdm
import pandas as pd
from JudgePrompt import generate_judge_prompt
from JudgePrompt import json_to_dataframe
import time
from FindTheBestModel import calculate_total_and_average_scores

client = AzureOpenAI(
    api_version="2024-02-15-preview",
    api_key="f8c3f26d14ff4876ab9a7d23251337d5",
    azure_endpoint="https://foxbrainopenaiapieastus.openai.azure.com",
)

def llm_call(messages, model="gpt-4o"):
    response = client.chat.completions.create(model=model, temperature=0.0, top_p=1, messages=messages)
    return response.choices[0].message.content

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def read_existing_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def write_json(data, file_path):
    existing_data = read_existing_json(file_path)
    existing_data.append(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

def evaluate_text(original_text, rewritten_text, max_retries=3):
    judge_prompt = generate_judge_prompt(original_text, rewritten_text)
    messages = [{"role": "user", "content": judge_prompt}]
    
    for attempt in range(max_retries):
        try:
            evaluation_response = llm_call(messages)
            evaluation_result = json.loads(evaluation_response)
            return evaluation_result
        except json.JSONDecodeError as e:
            print(f"JSON 解碼錯誤: {e}")
            print("重試中...")
            time.sleep(1)  # 等待一秒後重試

    print("超過最大重試次數，無法獲得有效的 JSON 格式響應。")
    return None

def process_news(original_text, rewritten_texts, output_file_path):
    result = {}
    for model, rewritten_text in rewritten_texts.items():
        evaluation_result = evaluate_text(original_text, rewritten_text)
        result[model] = evaluation_result
    write_json(result, output_file_path)

def main():
    original_texts_file_path = '../data/example.jsonl'
    rewrite_files = {
        'qwen': '../RewriteResults/...',
        'llama': '../RewriteResults/...',
        'yi': '../RewriteResults/....',
        'gemma2': '../RewriteResults/.....',
        'command': '../RewriteResults/....',
        'deepseek': '../RewriteResults/...'
    }

    original_texts = [item['content'] for item in read_jsonl(original_texts_file_path)]
    rewrites = {model: read_json(path) for model, path in rewrite_files.items()}
    
    print("步驟 1：生成 evaluation_result.json")
    for i, original_text in tqdm(enumerate(original_texts), desc="Processing news items"):
        rewritten_texts = {model: rewrites[model][i] for model in rewrites}
        process_news(original_text, rewritten_texts, 'evaluation_result.json')

    print("步驟 2：讀取 evaluation_result.json")
    input_path = 'evaluation_result.json'
    json_data = read_json(input_path)
    df = json_to_dataframe(json_data)
    
    print("步驟 3：將結果寫入 evaluation_result.csv")
    csv_file_path = 'evaluation_result.csv'
    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    print("所有步驟完成")

    print("步驟 4：計算模型總分和平均分")
    results_df = calculate_total_and_average_scores(df)
    results_df.to_csv('model_performance.csv', index=False, encoding='utf-8-sig')
    print("所有步驟完成")

if __name__ == "__main__":
    main()
