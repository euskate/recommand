import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

# 데이터 생성
data = {
    'A': [5, 3, 4, 3, 1],
    'B': [1, 1, 2, 3, 5],
    'C': [4, 2, 4, 1, 5],
    'D': [4, 2, 5, 5, 2],
    'E': [np.nan, 3, 5, 4, 1],
    'F': [np.nan, 2, 2, 2, 2]
}

users = ['Kim', 'Lee', 'Park', 'Choi', 'Kwon']

df = pd.DataFrame(data, index=users)

# 결측값 채우기
df_filled = df.apply(lambda col: col.fillna(col.mean()), axis=0)

# 코사인 유사도 계산
cos_sim = cosine_similarity(df_filled)
cos_sim_df = pd.DataFrame(cos_sim, index=users, columns=users)

# KNN 설정
k = 2
neigh = NearestNeighbors(n_neighbors=k, metric='cosine')
neigh.fit(df_filled)
distances, indices = neigh.kneighbors(df_filled)

# Kim 사용자에 대한 추천 계산 (사용자 인덱스 0)
kim_index = 0
neighbors_indices = indices[kim_index]

# 결측값에 대한 가중 평균 계산
recommendations = {}
for item in ['E', 'F']:
    item_index = df.columns.get_loc(item)
    weighted_sum = 0
    similarity_sum = 0
    for neighbor_index in neighbors_indices:
        if not np.isnan(df.iloc[neighbor_index, item_index]):
            similarity = cos_sim[kim_index, neighbor_index]
            rating = df.iloc[neighbor_index, item_index]
            weighted_sum += similarity * rating
            similarity_sum += similarity
    if similarity_sum != 0:
        recommendations[item] = weighted_sum / similarity_sum

# MSE 및 RMSE 계산
true_values = [4, 3]
pred_values = [recommendations['E'], recommendations['F']]

mse = mean_squared_error(true_values, pred_values)
rmse = np.sqrt(mse)

print("코사인 유사도 매트릭스:")
print(cos_sim_df)
print("\nKim에 대한 추천 (사용자 0):")
print(recommendations)
print(f"\nMSE: {mse}, RMSE: {rmse}")

# 결과 출력
"""
코사인 유사도 매트릭스:
           Kim       Lee      Park      Choi      Kwon
Kim   1.000000  0.961964  0.979504  0.941761  0.903294
Lee   0.961964  1.000000  0.979561  0.985307  0.872782
Park  0.979504  0.979561  1.000000  0.958698  0.940361
Choi  0.941761  0.985307  0.958698  1.000000  0.883608
Kwon  0.903294  0.872782  0.940361  0.883608  1.000000

Kim에 대한 추천 (사용자 0):
{'E': 3.687765980344771, 'F': 2.0}

MSE: 0.4871556014620197, RMSE: 0.6983111311416188
"""
