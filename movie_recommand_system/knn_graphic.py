import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# 사용자 데이터
data = {
    'A': [5, 3, 4, 3, 1],
    'B': [1, 1, 2, 3, 5],
    'C': [4, 2, 4, 1, 5],
    'D': [4, 2, 5, 5, 2],
    'E': [np.nan, 3, 5, 4, 1],
    'F': [np.nan, 2, 2, 3, 4]
}

users = ['kim', 'lee', 'park', 'choi', 'kwon']

# 데이터프레임 생성
df = pd.DataFrame(data, index=users)

# 결측값을 각 열의 평균으로 채우기
df_filled = df.apply(lambda col: col.fillna(col.mean()), axis=0)

# 코사인 유사도 계산
cos_sim = cosine_similarity(df_filled)

# KNN 설정
k = 2  # 예를 들어 k=2로 설정
neigh = NearestNeighbors(n_neighbors=k, metric='cosine')
neigh.fit(df_filled)

# Kim의 이웃 찾기 (사용자 인덱스 0)
distances, indices = neigh.kneighbors(df_filled)

# Kim의 이웃 인덱스 추출
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

# Kim의 결측값(E와 F)에 대한 예측 결과 출력
print(f"Kim의 E에 대한 예측값: {recommendations['E']}")
print(f"Kim의 F에 대한 예측값: {recommendations['F']}")

# 결과 그래프
labels = ['A', 'B', 'C', 'D', 'E', 'F']
kim_values = [5, 1, 4, 4, recommendations['E'], recommendations['F']]

plt.figure(figsize=(10, 6))
plt.bar(labels, kim_values, color='skyblue')
plt.title("Kim's Predicted Ratings for E and F")
plt.xlabel("Items")
plt.ylabel("Ratings")
plt.ylim(0, 6)
plt.show()
