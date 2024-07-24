import pandas as pd
from FindTheBestModel import calculate_total_and_average_scores

def generate_judge_prompt(original_text, rewritten_text):
    return f"""
    ### Role
    You are an expert judge responsible for evaluating the quality of rewritten articles. Each time you score, it must be objective and consistent.

    ### Task
    Compare all the prompts I provide, analyze each one, and compare which one better meets the requirements of the prompt, explaining the reasons.

    ### Goal: Users will input articles modified based on the original text. Please carefully evaluate the given answers according to the following criteria and steps, and finally provide a rating.

    ### Style: Please use a professional and easy-to-understand language style, with clear and structured organization.

    ### Analysis Prompt: Analyze each prompt according to the following evaluation criteria, score each criterion from 0 to 10, and explain in detail (1) why you gave this score and (2) the direction for modification.

    ### Requirements or Responses: Evaluation criteria (total score, 0-100, can include decimals):

    Here is a detailed description of each score, using a range of 1 to 10 points:

    #### 1. Clarity (1-10 points)
    - 1 point: The rewritten sentence is more obscure and difficult to understand.
    - 2 points: The rewritten sentence is slightly improved but still difficult to understand.
    - 3 points: The rewritten sentence is somewhat improved but still vague.
    - 4 points: The rewritten sentence is significantly improved but could be clearer.
    - 5 points: The rewritten sentence is relatively simple and clear but could still be clearer.
    - 6 points: The rewritten sentence is simple and easy to understand, with only a few sentences that could be further simplified.
    - 7 points: The rewritten sentence is very clear, but there is still a slight room for improvement.
    - 8 points: The rewritten sentence is very clear, with almost no need for further simplification.
    - 9 points: The rewritten sentence is extremely clear, with only minor adjustments needed.
    - 10 points: The rewritten sentence is completely simple and clear, with no further modifications needed.

    #### 2. Retention of Original Meaning (1-10 points)
    - 1 point: The rewriting process completely lost the original meaning.
    - 2 points: The rewritten text lost most of the original meaning, retaining only a small part.
    - 3 points: The rewritten text retains some of the original meaning but still lost most of it.
    - 4 points: The rewritten text retains about half of the original meaning.
    - 5 points: The rewritten text retains most of the original meaning, but some are still lost.
    - 6 points: The rewritten text retains most of the original meaning, with only a few details lost.
    - 7 points: The rewritten text basically retains the original meaning, with only minor parts lost.
    - 8 points: The rewritten text basically retains the original meaning, with only minor details lost.
    - 9 points: The rewritten text retains almost all of the original meaning, with only minor improvements needed.
    - 10 points: The rewritten text completely retains the original meaning and is expressed more clearly.

    #### 3. Descriptiveness (1-10 points)
    - 1 point: Almost no additional descriptive details were added.
    - 2 points: Some details were added but not rich enough.
    - 3 points: A small amount of descriptive details were added but still need more.
    - 4 points: There are some descriptive details, but they are not rich enough.
    - 5 points: There is an appropriate amount of descriptive details, but more can still be added.
    - 6 points: More descriptive details were added, but there is still room for improvement.
    - 7 points: There are rich descriptive details, but some minor details can still be added.
    - 8 points: Rich descriptive details were added, with only a few minor details to be supplemented.
    - 9 points: There are ample and rich descriptive details, with only minor supplements needed.
    - 10 points: There are rich and sufficient descriptive details, making the content more vivid, with no need for further supplements.

    #### 4. Depth (1-10 points)
    - 1 point: The added details provide no in-depth insights.
    - 2 points: The added details provide some insights but are not deep enough.
    - 3 points: The added details provide a small amount of insights but are still not deep enough.
    - 4 points: The added details provide some insights, but the depth is limited.
    - 5 points: The added details provide an appropriate amount of insights but are not deep enough.
    - 6 points: The added details provide more insights but still have room for improvement.
    - 7 points: The added details provide in-depth insights but still have further exploration space.
    - 8 points: The added details provide more in-depth insights, with only minor space for further exploration.
    - 9 points: The added details provide profound insights, with only minor adjustments needed.
    - 10 points: The added details provide profound and comprehensive insights, making the paragraph deeper.

    #### 5. Diversity of Perspectives (1-10 points)
    - 1 point: No opposing opinions or counterarguments were introduced.
    - 2 points: Only a few opposing opinions or counterarguments were introduced.
    - 3 points: Some opposing opinions were introduced but not enough.
    - 4 points: Some opposing opinions were introduced, but they are not rich enough.
    - 5 points: Some opposing opinions or counterarguments were introduced but more are needed.
    - 6 points: Various opposing opinions were introduced but still have room for supplementation.
    - 7 points: Various opposing opinions or counterarguments were introduced, with only minor viewpoints that can be supplemented.
    - 8 points: Rich opposing opinions or counterarguments were introduced, with only minor insufficiencies.
    - 9 points: Rich opposing opinions or counterarguments were introduced, making the viewpoints diverse, with only minor space for supplementation.
    - 10 points: Rich and comprehensive opposing opinions or counterarguments were introduced, making the viewpoints diverse and comprehensive.

    #### 6. Balance (1-10 points)
    - 1 point: The viewpoints are severely unbalanced, biased towards one side.
    - 2 points: The viewpoints have obvious bias.
    - 3 points: The viewpoints are somewhat biased, not balanced.
    - 4 points: The viewpoints have slight bias but are generally balanced.
    - 5 points: The viewpoints are basically balanced but have minor bias.
    - 6 points: The viewpoints are relatively balanced but still have minor deviations.
    - 7 points: The viewpoints are generally balanced, with only minor bias.
    - 8 points: The viewpoints are almost completely balanced, with only minor bias.
    - 9 points: The viewpoints are almost completely balanced, with only minor adjustments needed.
    - 10 points: The viewpoints are completely balanced, with no bias.

    #### 7. Vocabulary Diversity (1-10 points)
    - 1 point: Vocabulary is repetitive, with no diversity.
    - 2 points: Only a few synonyms were replaced.
    - 3 points: Some synonyms were replaced, but diversity is insufficient.
    - 4 points: Some synonyms were replaced but still have room for improvement.
    - 5 points: Some synonyms were replaced but vocabulary diversity still needs improvement.
    - 6 points: More synonyms were replaced, making the vocabulary diverse.
    - 7 points: A large number of synonyms were replaced, making the vocabulary rich in diversity.
    - 8 points: A wide range of synonyms were replaced, making the vocabulary rich in diversity, with only minor additions needed.
    - 9 points: Vocabulary diversity is very rich, with only minor improvements needed.
    - 10 points: Vocabulary diversity is extremely rich, with no further replacements needed.

    #### 8. Meaning Preservation (1-10 points)
    - 1-2 points: The meaning was completely lost after replacing the vocabulary.
    - 3-4 points: Most of the original meaning was lost after replacing the vocabulary.
    - 5-6 points: About half of the original meaning was preserved after replacing the vocabulary.
    - 7-8 points: Most of the original meaning was preserved after replacing the vocabulary.
    - 9-10 points: The original meaning was completely preserved after replacing the vocabulary, and the expression is smoother.

    #### 9. Focus (1-10 points)
    - 1 point: The rewritten text deviates from the theme.
    - 2 points: The rewritten text partially deviates from the theme.
    - 3 points: The rewritten text has obvious deviations from the theme.
    - 4 points: The rewritten text has some deviations from the theme.
    - 5 points: The rewritten text is basically focused on the theme but has some deviations.
    - 6 points: The rewritten text is relatively focused on the theme but has minor deviations.
    - 7 points: The rewritten text is generally focused on the theme, with only minor deviations.
    - 8 points: The rewritten text is almost completely focused on the theme, with only minor deviations.
    - 9 points: The rewritten text is completely focused on the theme, with only minor adjustments needed.
    - 10 points: The rewritten text is completely focused on the specific theme or aspect, with no deviations.

    #### 10. Emphasis (1-10 points)
    - 1 point: Failed to emphasize any points successfully.
    - 2 points: Only a few points were emphasized.
    - 3 points: Some points were emphasized

 but not obvious enough.
    - 4 points: Some points were emphasized but could be more prominent.
    - 5 points: Most points were emphasized but could still be more prominent.
    - 6 points: Most points were emphasized, making them clear.
    - 7 points: Successfully emphasized most points but still have room for improvement.
    - 8 points: Successfully emphasized specific points, making them more prominent, with minor improvements needed.
    - 9 points: All important points were emphasized, with only minor adjustments needed.
    - 10 points: Successfully emphasized all specific points, making them extremely prominent, with no further emphasis needed.

    ### Evaluation Steps:
    1. Analyze each prompt according to the above criteria, score each criterion from 0 to 10, and explain in detail why you gave this score and the direction for modification.

    Please make sure to provide your evaluation in the following JSON format: ("DO NOT" output using ```json)
    {{
      "Simplify for Different Reading Levels": {{
        "Clarity": {{
          "score": score,
          "explanation": "Your explanation here"
        }},
        "Retention of Original Meaning": {{
          "score": score,
          "explanation": "Your explanation here"
        }}
      }},
      "Enhance Details": {{
        "Descriptiveness": {{
          "score": score,
          "explanation": "Your explanation here"
        }},
        "Depth": {{
          "score": score,
          "explanation": "Your explanation here"
        }}
      }},
      "Contrast Viewpoints": {{
        "Diversity of Perspectives": {{
          "score": score,
          "explanation": "Your explanation here"
        }},
        "Balance": {{
          "score": score,
          "explanation": "Your explanation here"
        }}
      }},
      "Add Synonyms": {{
        "Vocabulary Diversity": {{
          "score": score,
          "explanation": "Your explanation here"
        }},
        "Meaning Preservation": {{
          "score": score,
          "explanation": "Your explanation here"
        }}
      }},
      "Theme Rewriting": {{
        "Focus": {{
          "score": score,
          "explanation": "Your explanation here"
        }},
        "Emphasis": {{
          "score": score,
          "explanation": "Your explanation here"
        }}
      }},
      "overall_score": total_score,
      "overall_feedback": "Your explanation here"
    }}

    ### Constraints:
    (1) Please respond in Traditional Chinese.
    (2) Please maintain objective fairness, and each time you score, it must be objective and consistent.

    Now, please evaluate the following content, remember to reply in the specified format, and each time you score, it must be objective and consistent:

    Original Text:
    {original_text}

    Rewritten Text:
    {rewritten_text}
    """
import pandas as pd

def json_to_dataframe(json_data):
    records = []
    for news_id, models in enumerate(json_data):
        for model, evaluation_result in models.items():
            if evaluation_result is None:
                evaluation_result = {}
            
            def get_nested_value(data, keys, default_value):
                for key in keys:
                    if key in data:
                        data = data[key]
                    else:
                        return default_value
                return data

            record = {
                "NewsID": news_id,
                "Model": model,
                "清晰度_score": get_nested_value(evaluation_result, ['Simplify for Different Reading Levels', 'Clarity', 'score'], 0),
                "清晰度_explanation": get_nested_value(evaluation_result, ['Simplify for Different Reading Levels', 'Clarity', 'explanation'], '無法獲得有效評估'),
                "保留原意_score": get_nested_value(evaluation_result, ['Simplify for Different Reading Levels', 'Retention of Original Meaning', 'score'], 0),
                "保留原意_explanation": get_nested_value(evaluation_result, ['Simplify for Different Reading Levels', 'Retention of Original Meaning', 'explanation'], '無法獲得有效評估'),
                "描述性_score": get_nested_value(evaluation_result, ['Enhance Details', 'Descriptiveness', 'score'], 0),
                "描述性_explanation": get_nested_value(evaluation_result, ['Enhance Details', 'Descriptiveness', 'explanation'], '無法獲得有效評估'),
                "深度_score": get_nested_value(evaluation_result, ['Enhance Details', 'Depth', 'score'], 0),
                "深度_explanation": get_nested_value(evaluation_result, ['Enhance Details', 'Depth', 'explanation'], '無法獲得有效評估'),
                "觀點多樣性_score": get_nested_value(evaluation_result, ['Contrast Viewpoints', 'Diversity of Perspectives', 'score'], 0),
                "觀點多樣性_explanation": get_nested_value(evaluation_result, ['Contrast Viewpoints', 'Diversity of Perspectives', 'explanation'], '無法獲得有效評估'),
                "平衡性_score": get_nested_value(evaluation_result, ['Contrast Viewpoints', 'Balance', 'score'], 0),
                "平衡性_explanation": get_nested_value(evaluation_result, ['Contrast Viewpoints', 'Balance', 'explanation'], '無法獲得有效評估'),
                "詞彙多樣性_score": get_nested_value(evaluation_result, ['Add Synonyms', 'Vocabulary Diversity', 'score'], 0),
                "詞彙多樣性_explanation": get_nested_value(evaluation_result, ['Add Synonyms', 'Vocabulary Diversity', 'explanation'], '無法獲得有效評估'),
                "意思保留_score": get_nested_value(evaluation_result, ['Add Synonyms', 'Meaning Preservation', 'score'], 0),
                "意思保留_explanation": get_nested_value(evaluation_result, ['Add Synonyms', 'Meaning Preservation', 'explanation'], '無法獲得有效評估'),
                "聚焦性_score": get_nested_value(evaluation_result, ['Theme Rewriting', 'Focus', 'score'], 0),
                "聚焦性_explanation": get_nested_value(evaluation_result, ['Theme Rewriting', 'Focus', 'explanation'], '無法獲得有效評估'),
                "強調點_score": get_nested_value(evaluation_result, ['Theme Rewriting', 'Emphasis', 'score'], 0),
                "強調點_explanation": get_nested_value(evaluation_result, ['Theme Rewriting', 'Emphasis', 'explanation'], '無法獲得有效評估'),
                "overall_score": evaluation_result.get('overall_score', 0),
                "overall_feedback": evaluation_result.get('overall_feedback', '無法獲得有效評估')
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    return df
    
    
#from litellm import completion, embedding
#from openai import AzureOpenAI
#import json
#import os
#from tqdm import tqdm
#import pandas as pd
#from JudgePrompt import generate_judge_prompt
#from JudgePrompt import json_to_dataframe
#import time
#    
#def read_file(file_path):
#    with open(file_path, 'r', encoding='utf-8') as file:
#        return file.read()
#
#def read_json(file_path):
#    with open(file_path, 'r', encoding='utf-8') as f:
#        return json.load(f)
#
#def read_jsonl(file_path):
#    with open(file_path, 'r', encoding='utf-8') as f:
#        return [json.loads(line) for line in f]
#
#def read_existing_json(file_path):
#    if os.path.exists(file_path):
#        with open(file_path, 'r', encoding='utf-8') as f:
#            try:
#                return json.load(f)
#            except json.JSONDecodeError:
#                return []
#    return []
#
#def write_json(data, file_path):
#    with open(file_path, 'w', encoding='utf-8') as f:
#        json.dump(data, f, ensure_ascii=False, indent=4)
#import os
#def jsonToCsv(output_dir,category):
#    output_file_path = os.path.join(output_dir, f'{category}_result.json')
#    # 生成每個分類的 evaluation_result.csv 和 model_performance.csv
#    json_data = read_json(output_file_path)
#    df = json_to_dataframe(json_data)
#    csv_file_path = os.path.join(output_dir, f'{category}_evaluation_result.csv')
#    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
#
#    results_df = calculate_total_and_average_scores(df)
#    results_df.to_csv(os.path.join(output_dir, f'{category}_model_performance.csv'), index=False, encoding='utf-8-sig')
#
##    # Mark category as completed
##    progress[category] = 'completed'
##    save_progress(progress, progress_file)
#
#output_dir='../EvaluateResultsTest'
#category='Jobs_and_Education'
#jsonToCsv(output_dir,category)
    
#
#def json_to_dataframe(json_data):
#    records = []
#    for news_id, models in enumerate(json_data):
#        for model, evaluation_result in models.items():
#            if evaluation_result is None:
#                evaluation_result = {
#                    "簡化以適應不同閱讀水平": {
#                        "清晰度": {"score": 0, "explanation": "無法獲得有效評估"},
#                        "保留原意": {"score": 0, "explanation": "無法獲得有效評估"}
#                    },
#                    "增強細節": {
#                        "描述性": {"score": 0, "explanation": "無法獲得有效評估"},
#                        "深度": {"score": 0, "explanation": "無法獲得有效評估"}
#                    },
#                    "對比觀點": {
#                        "觀點多樣性": {"score": 0, "explanation": "無法獲得有效評估"},
#                        "平衡性": {"score": 0, "explanation": "無法獲得有效評估"}
#                    },
#                    "加入同義詞": {
#                        "詞彙多樣性": {"score": 0, "explanation": "無法獲得有效評估"},
#                        "意思保留": {"score": 0, "explanation": "無法獲得有效評估"}
#                    },
#                    "主題重寫": {
#                        "聚焦性": {"score": 0, "explanation": "無法獲得有效評估"},
#                        "強調點": {"score": 0, "explanation": "無法獲得有效評估"}
#                    },
#                    "overall_score": 0,
#                    "overall_feedback": "無法獲得有效評估"
#                }
#            record = {
#                "NewsID": news_id,
#                "Model": model,
#                "清晰度_score": evaluation_result['簡化以適應不同閱讀水平']['清晰度']['score'],
#                "清晰度_explanation": evaluation_result['簡化以適應不同閱讀水平']['清晰度']['explanation'],
#                "保留原意_score": evaluation_result['簡化以適應不同閱讀水平']['保留原意']['score'],
#                "保留原意_explanation": evaluation_result['簡化以適應不同閱讀水平']['保留原意']['explanation'],
#                "描述性_score": evaluation_result['增強細節']['描述性']['score'],
#                "描述性_explanation": evaluation_result['增強細節']['描述性']['explanation'],
#                "深度_score": evaluation_result['增強細節']['深度']['score'],
#                "深度_explanation": evaluation_result['增強細節']['深度']['explanation'],
#                "觀點多樣性_score": evaluation_result['對比觀點']['觀點多樣性']['score'],
#                "觀點多樣性_explanation": evaluation_result['對比觀點']['觀點多樣性']['explanation'],
#                "平衡性_score": evaluation_result['對比觀點']['平衡性']['score'],
#                "平衡性_explanation": evaluation_result['對比觀點']['平衡性']['explanation'],
#                "詞彙多樣性_score": evaluation_result['加入同義詞']['詞彙多樣性']['score'],
#                "詞彙多樣性_explanation": evaluation_result['加入同義詞']['詞彙多樣性']['explanation'],
#                "意思保留_score": evaluation_result['加入同義詞']['意思保留']['score'],
#                "意思保留_explanation": evaluation_result['加入同義詞']['意思保留']['explanation'],
#                "聚焦性_score": evaluation_result['主題重寫']['聚焦性']['score'],
#                "聚焦性_explanation": evaluation_result['主題重寫']['聚焦性']['explanation'],
#                "強調點_score": evaluation_result['主題重寫']['強調點']['score'],
#                "強調點_explanation": evaluation_result['主題重寫']['強調點']['explanation'],
#                "overall_score": evaluation_result['overall_score'],
#                "overall_feedback": evaluation_result['overall_feedback']
#            }
#            records.append(record)
#    
#    df = pd.DataFrame(records)
#    return df
