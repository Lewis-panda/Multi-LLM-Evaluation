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
    api_key="Your Key",
    azure_endpoint="Your Endpoint",
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
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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

def process_news(original_text, rewritten_texts):
    result = {}
    for model, rewritten_text in tqdm(rewritten_texts.items(), desc="Processing models"):
        evaluation_result = evaluate_text(original_text, rewritten_text)
        result[model] = evaluation_result
    return result

def save_progress(progress, progress_file):
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def process_category(model_dirs, category_dirs, output_dir, progress_file='../EvaluateResults/ds.pos'):
    progress = load_progress(progress_file)

    for category in tqdm(category_dirs, desc="Processing categories"):
        if category in progress and progress[category] == 'completed':
            continue

        original_file_path = os.path.join('../classify_data', category, 'first50.jsonl')
        original_texts = [item['text'] for item in read_jsonl(original_file_path)]

        rewritten_texts_per_category = {}
        for model, model_dir in model_dirs.items():
            rewrite_file_path = os.path.join(model_dir, category, 'first50', 'first50.json')
            rewritten_texts_per_category[model] = read_json(rewrite_file_path)

        start_index = progress.get(category, 0)
        results = []
        for i in tqdm(range(start_index, len(original_texts)), desc=f"Processing texts in {category}"):
            original_text = original_texts[i]
            rewritten_texts_per_doc = {model: rewritten_texts_per_category[model][i] for model in model_dirs}
            result = process_news(original_text, rewritten_texts_per_doc)
            results.append(result)

            # Update progress after each text
            progress[category] = i + 1
            save_progress(progress, progress_file)

        category_output_dir = os.path.join(output_dir, category)
        if not os.path.exists(category_output_dir):
            os.makedirs(category_output_dir)

        output_file_path = os.path.join(category_output_dir, 'result.json')
        write_json(results, output_file_path)

        # 生成每個分類的 evaluation_result.csv 和 model_performance.csv
        json_data = read_json(output_file_path)
        df = json_to_dataframe(json_data)
        csv_file_path = os.path.join(category_output_dir, 'evaluation_result.csv')
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

        results_df = calculate_total_and_average_scores(df)
        results_df.to_csv(os.path.join(category_output_dir, 'model_performance.csv'), index=False, encoding='utf-8-sig')

        # Mark category as completed
        progress[category] = 'completed'
        save_progress(progress, progress_file)

# For testing
def main():
    model_dirs = {
        'qwen': '../RewriteResults/qwen2',
        'llama': '../RewriteResults/llama3',
        'yi': '../RewriteResults/yi',
        'gemma2': '../RewriteResults/gemma2',
        'command': '../RewriteResults/command'
#        'deepseek': '../RewriteResults/deepseek'
    }

    category_dirs = [d for d in os.listdir('../classify_data/') if os.path.isdir(os.path.join('../classify_data/', d))]
    
    output_dir = '../EvaluateResults'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_category(model_dirs, category_dirs, output_dir)

if __name__ == "__main__":
    main()
