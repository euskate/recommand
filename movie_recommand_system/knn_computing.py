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

# 코사인 유사도 행렬 출력
cos_sim_df = pd.DataFrame(cos_sim, index=users, columns=users)
print("코사인 유사도 행렬:")
print(cos_sim_df)

# KNN 설정
k = 2  # 예를 들어 k=2로 설정
neigh = NearestNeighbors(n_neighbors=k, metric='cosine')
neigh.fit(df_filled)

# Kim의 이웃 찾기 (사용자 인덱스 0)
distances, indices = neigh.kneighbors([df_filled.loc['kim']])

# Kim의 이웃 인덱스 추출 및 거리 출력
print("\nKim의 KNN 이웃 인덱스와 거리:")
for i, index in enumerate(indices[0]):
    print(f"이웃 {i+1}: 사용자 '{users[index]}' (거리: {distances[0][i]})")

# 결측값에 대한 가중 평균 계산
recommendations = {}
for item in ['E', 'F']:
    item_index = df.columns.get_loc(item)
    weighted_sum = 0
    similarity_sum = 0
    for neighbor_index in indices[0]:
        if not np.isnan(df.iloc[neighbor_index, item_index]):
            similarity = 1 - distances[0][indices[0].tolist().index(neighbor_index)]  # 코사인 거리 -> 유사도로 변환
            rating = df.iloc[neighbor_index, item_index]
            weighted_sum += similarity * rating
            similarity_sum += similarity
            print(f"\n{item} 예측에 사용된 '{users[neighbor_index]}'의 평점: {rating} (유사도: {similarity})")
    if similarity_sum != 0:
        recommendations[item] = weighted_sum / similarity_sum
        print(f"{item}의 예측 값: {recommendations[item]}")

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

# 결과
"""
코사인 유사도 행렬:
           kim       lee      park      choi      kwon
kim   1.000000  0.972652  0.963487  0.872715  0.705759
lee   0.972652  1.000000  0.965535  0.908121  0.698501
park  0.963487  0.965535  1.000000  0.926354  0.708088
choi  0.872715  0.908121  0.926354  1.000000  0.695193
kwon  0.705759  0.698501  0.708088  0.695193  1.000000

Kim의 KNN 이웃 인덱스와 거리:
이웃 1: 사용자 'kim' (거리: 0.0)
이웃 2: 사용자 'lee' (거리: 0.02734838304539633)

E 예측에 사용된 'lee'의 평점: 3.0 (유사도: 0.9726516169546037)
E의 예측 값: 3.0

F 예측에 사용된 'lee'의 평점: 2.0 (유사도: 0.9726516169546037)
F의 예측 값: 2.0
"""
