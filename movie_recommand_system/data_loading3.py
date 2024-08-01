import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 파일 경로 설정
file_path = 'kim.txt'

# 파일을 읽고 데이터를 리스트로 변환합니다.
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 각 줄을 쉼표로 분리
        parts = line.strip().split(',')
        if len(parts) == 3:  # 항목이 3개인지 확인
            user, item, rating = parts
            try:
                rating = int(rating)  # 정수로 변환 시도
                data.append([user, item, rating])
            except ValueError:
                print(f"Warning: Invalid rating value '{rating}' for user '{user}' and item '{item}'. Skipping.")

# 리스트를 데이터프레임으로 변환합니다.
df = pd.DataFrame(data, columns=['User', 'Item', 'Rating'])

# 피벗 테이블을 생성하여 사용자-아이템 매트릭스로 변환합니다.
pivot_table = df.pivot_table(index='User', columns='Item', values='Rating', fill_value=0)

# 피벗 테이블의 사용자 벡터 간의 코사인 유사도 계산
cosine_sim_matrix = cosine_similarity(pivot_table)
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=pivot_table.index, columns=pivot_table.index)


# 코사인 유사도 계산 함수
def predict_rating(user, item, pivot_table, cosine_sim_df):
    if item not in pivot_table.columns:
        return np.nan  # 아이템이 없으면 예측할 수 없음

    similar_users = cosine_sim_df[user]
    item_ratings = pivot_table[item]

    # 유사도가 0인 사용자 제외
    non_zero_similar_users = similar_users[item_ratings > 0]
    non_zero_item_ratings = item_ratings[item_ratings > 0]

    if len(non_zero_similar_users) == 0:
        return np.nan  # 유사한 사용자가 없으면 예측할 수 없음

    weighted_sum = np.sum(non_zero_similar_users * non_zero_item_ratings)
    sim_sum = np.sum(np.abs(non_zero_similar_users))

    if sim_sum == 0:
        return np.nan

    return weighted_sum / sim_sum


# 사용자 kim의 E와 F에 대한 평점 예측
user_to_predict = 'kim'
items_to_predict = ['E', 'F']

predictions = {}
for item in items_to_predict:
    predictions[item] = predict_rating(user_to_predict, item, pivot_table, cosine_sim_df)

# 예측 결과를 데이터프레임으로 변환합니다.
predictions_df = pd.DataFrame(list(predictions.items()), columns=['Item', 'PredictedRating'])

# 모든 사용자와 아이템 데이터와 예측 결과를 출력합니다.
print("User-Item Matrix:")
print(pivot_table)
print("\nCosine Similarity Matrix:")
print(cosine_sim_df)
print("\nPredicted Ratings for 'kim':")
print(predictions_df)
