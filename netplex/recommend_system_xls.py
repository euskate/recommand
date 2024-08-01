import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 파일 경로 설정
file_path = 'u.data'
item_file_path = 'u.item'
output_excel_path = 'recommendations.xlsx'

# 데이터 파일을 읽어오고 데이터프레임으로 변환
df = pd.read_csv(file_path, sep='\t', header=None, encoding='utf-8')
df.columns = ['UserID', 'ItemID', 'Rating', 'Timestamp']

# 아이템 파일을 읽어오고 데이터프레임으로 변환
try:
    items_df = pd.read_csv(item_file_path, sep='|', header=None, encoding='ISO-8859-1', usecols=[0, 1],
                           names=['ItemID', 'Title'], engine='python')
except UnicodeDecodeError as e:
    print(f"Error decoding file with ISO-8859-1 encoding: {e}")

# UserID를 기준으로 피벗 테이블 생성
pivot_table = df.pivot_table(index='UserID', columns='ItemID', values='Rating', fill_value=0)

# 피벗 테이블의 행(row)과 열(column)로 코사인 유사도 계산
cosine_sim_matrix = cosine_similarity(pivot_table.T)

# 코사인 유사도 행렬을 데이터프레임으로 변환
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=pivot_table.columns, columns=pivot_table.columns)


def recommend_items(user_id, pivot_table, cosine_sim_df, items_df, num_recommendations=5):
    # 사용자가 평가한 아이템
    user_ratings = pivot_table.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index

    # 사용자 평가에 대한 아이템 유사도 점수 계산
    sim_scores = {}
    for item in unrated_items:
        # 평가된 아이템들에 대한 유사도 점수 계산
        rated_items = user_ratings[user_ratings > 0].index
        sim_sum = np.sum([cosine_sim_df.loc[item, rated_item] for rated_item in rated_items])
        weighted_sum = np.sum(
            [user_ratings[rated_item] * cosine_sim_df.loc[item, rated_item] for rated_item in rated_items])

        if sim_sum != 0:
            sim_scores[item] = weighted_sum / sim_sum

    # 추천 점수로 정렬
    sorted_items = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [item for item, score in sorted_items[:num_recommendations]]

    # 아이템 제목으로 변환
    recommendations_titles = items_df.set_index('ItemID').loc[recommendations]['Title'].tolist()

    return recommendations_titles


# 추천 결과를 저장할 데이터프레임 생성
results = []

# 모든 사용자에게 추천 아이템을 계산
for user_id in pivot_table.index:
    recommendations = recommend_items(user_id, pivot_table, cosine_sim_df, items_df)
    for title in recommendations:
        results.append({'UserID': user_id, 'RecommendedItem': title})

# 결과를 데이터프레임으로 변환
results_df = pd.DataFrame(results)

# Excel 파일로 저장
results_df.to_excel(output_excel_path, index=False)

print(f"추천 결과가 '{output_excel_path}' 파일에 저장되었습니다.")
