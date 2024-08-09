import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# 이미지에서 가져온 데이터
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

# 각 열의 평균으로 결측값 채우기
df_filled = df.apply(lambda col: col.fillna(col.mean()), axis=0)

# 코사인 유사도 계산
cos_sim = cosine_similarity(df_filled)
cos_sim_df = pd.DataFrame(cos_sim, index=users, columns=users)

# KNN 설정
k = 2
neigh = NearestNeighbors(n_neighbors=k, metric='cosine')
neigh.fit(df_filled)
distances, indices = neigh.kneighbors(df_filled)

# Kim에게 추천 계산 (사용자 인덱스 0)
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
# Kim의 E와 F의 실제 값이 수동으로 입력되었다고 가정
true_values = [4, 3]  # 임의 값이며, 실제 값이 있다면 해당 값으로 대체해야 합니다.
pred_values = [recommendations['E'], recommendations['F']]

mse = mean_squared_error(true_values, pred_values)
rmse = np.sqrt(mse)

print("코사인 유사도 매트릭스:")
print(cos_sim_df)
print("\nKim에 대한 추천 (사용자 0):")
print(recommendations)
print(f"\nMSE: {mse}, RMSE: {rmse}")

# MSE와 RMSE 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(['E', 'F'], true_values, label='True Values', marker='o', linestyle='--', color='blue')
plt.plot(['E', 'F'], pred_values, label='Predicted Values', marker='o', linestyle='-', color='red')

plt.title('True vs Predicted Values')
plt.xlabel('Items')
plt.ylabel('Rating')
plt.legend()
plt.grid(True)
plt.show()