import os
from langchain_community.llms import Ollama
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# 定義要測試的主機和模型
model_host_pairs = [
    ("qwen2:72b-instruct-q5_K_M", 'http://13.65.249.11:8880'),
    ("llama3:70b-instruct-q5_K_M", 'http://13.65.249.11:6665'),
    ("yi:34b-v1.5", 'http://13.65.249.11:8885'),
    ("gemma2:27b", 'http://13.65.249.11:8889'),
    ("command-r-plus", 'http://13.65.249.11:8887'),
    ("deepseek-v2:236b", 'http://13.65.249.11:6662')
]

# 測試提示
test_prompt = "給我一篇三百字的文章，介紹大語言模型"

def test_model(host, model, timeout=60, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            llm = Ollama(
                model=model,
                base_url=host,
                keep_alive=True,
                num_predict=10  # 減少預測字符數，僅做測試用
            )
            response = llm.invoke(test_prompt, timeout=timeout)
            return {"model": model, "host": host, "status": "success", "response": response}
        except requests.exceptions.Timeout:
            attempt += 1
            print(f"Timeout for model {model} on host {host}, attempt {attempt}/{retries}")
        except Exception as e:
            if attempt >= retries - 1:
                return {"model": model, "host": host, "status": "failure", "error": str(e)}
    return {"model": model, "host": host, "status": "failure", "error": "Max retries exceeded"}

all_results = []

with ThreadPoolExecutor(max_workers=6) as executor:
    future_to_model = {executor.submit(test_model, host, model): (host, model) for model, host in model_host_pairs}
    with tqdm(total=len(model_host_pairs), desc="Testing models and hosts") as pbar:
        for future in as_completed(future_to_model):
            host, model = future_to_model[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"model": model, "host": host, "status": "failure", "error": str(exc)}
            all_results.append(result)
            print(f"Testing model {model} on host {host} - Status: {result['status']}")
            pbar.update(1)  # 更新進度條

with open("model_test_results2.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(all_results)

print("Testing completed. Results saved to model_test_results2.json")
