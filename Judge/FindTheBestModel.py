import pandas as pd

def calculate_total_and_average_scores(df):
    models = df['Model'].unique()
    results = []

    for model in models:
        model_df = df[df['Model'] == model]
        total_score = model_df[['清晰度_score', '保留原意_score', '描述性_score', '深度_score',
                                '觀點多樣性_score', '平衡性_score', '詞彙多樣性_score', '意思保留_score',
                                '聚焦性_score', '強調點_score']].sum(axis=1).sum()
        average_score = model_df[['清晰度_score', '保留原意_score', '描述性_score', '深度_score',
                                  '觀點多樣性_score', '平衡性_score', '詞彙多樣性_score', '意思保留_score',
                                  '聚焦性_score', '強調點_score']].sum(axis=1).mean()
        results.append({
            'Model': model,
            'Total Score': total_score,
            'Average Score': average_score
        })

    return pd.DataFrame(results)

def main():
    df = pd.read_csv('evaluation_result.csv', encoding='utf-8-sig')

    results_df = calculate_total_and_average_scores(df)
    
    results_df.to_csv('model_performance.csv', index=False, encoding='utf-8-sig')
    
    print(results_df)

if __name__ == "__main__":
    main()
