import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# 1. 파일 읽기 및 데이터 로딩
file_path = 'kim.txt'
data = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(',')
        user, item, rating = parts[0], parts[1], parts[2]
        if rating == '':
            rating = np.nan
        else:
            rating = float(rating)
        data.append((user, item, rating))

# 데이터프레임으로 변환
df = pd.DataFrame(data, columns=['User', 'Item', 'Rating'])
df_pivot = df.pivot(index='User', columns='Item', values='Rating')

# 데이터프레임 인덱스 확인
print("데이터프레임 인덱스:")
print(df_pivot.index)

# 2. 결측값을 각 열의 평균으로 채우기
df_filled = df_pivot.apply(lambda col: col.fillna(col.mean()), axis=0)

# 3. 코사인 유사도 계산
cos_sim = cosine_similarity(df_filled)

# 코사인 유사도 행렬 출력
users = df_pivot.index.tolist()
cos_sim_df = pd.DataFrame(cos_sim, index=users, columns=users)
print("코사인 유사도 행렬:")
print(cos_sim_df)

# 4. KNN 설정
k = 2  # K값 설정 (예시로 2 설정)
neigh = NearestNeighbors(n_neighbors=k, metric='cosine')
neigh.fit(df_filled)

# Kim의 인덱스 확인
if 'Kim' in df_filled.index:
    kim_index = 'Kim'
elif 'kim' in df_filled.index:
    kim_index = 'kim'
else:
    raise KeyError("'Kim' 또는 'kim'이라는 인덱스가 존재하지 않습니다.")

# Kim의 이웃 찾기 (사용자 인덱스 0)
distances, indices = neigh.kneighbors([df_filled.loc[kim_index]])

# Kim의 이웃 인덱스 추출 및 거리 출력
print("\nKim의 KNN 이웃 인덱스와 거리:")
for i, index in enumerate(indices[0]):
    print(f"이웃 {i+1}: 사용자 '{users[index]}' (거리: {distances[0][i]})")

# 5. 결측값에 대한 가중 평균 계산
recommendations = {}
for item in ['E', 'F']:
    item_index = df_filled.columns.get_loc(item)
    weighted_sum = 0
    similarity_sum = 0
    for neighbor_index in indices[0]:
        if not np.isnan(df_filled.iloc[neighbor_index, item_index]):
            similarity = 1 - distances[0][indices[0].tolist().index(neighbor_index)]  # 코사인 거리 -> 유사도로 변환
            rating = df_filled.iloc[neighbor_index, item_index]
            weighted_sum += similarity * rating
            similarity_sum += similarity
            print(f"\n{item} 예측에 사용된 '{users[neighbor_index]}'의 평점: {rating} (유사도: {similarity})")
    if similarity_sum != 0:
        recommendations[item] = weighted_sum / similarity_sum
        print(f"{item}의 예측 값: {recommendations[item]}")

# 6. 결과 그래프
labels = ['A', 'B', 'C', 'D', 'E', 'F']
kim_values = [
    df_filled.loc[kim_index, 'A'],
    df_filled.loc[kim_index, 'B'],
    df_filled.loc[kim_index, 'C'],
    df_filled.loc[kim_index, 'D'],
    recommendations.get('E', np.nan),
    recommendations.get('F', np.nan)
]

plt.figure(figsize=(10, 6))
plt.bar(labels, kim_values, color='skyblue')
plt.title("Kim's Predicted Ratings for E and F")
plt.xlabel("Items")
plt.ylabel("Ratings")
plt.ylim(0, 6)
plt.show()
