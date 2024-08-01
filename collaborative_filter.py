# 코사인 유사도 계산
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 사용자 벡터 정의 (길이 조정, 0으로 패딩)
kim = np.array([5, 1, 4, 4])
lee = np.array([3, 1, 2, 2])
park = np.array([4, 2, 4, 5])
choi = np.array([3, 3, 1, 5])
kwon = np.array([1, 5, 5, 2])

# 모든 사용자 벡터를 모아서 행렬로 만듦
users = np.array([kim, lee, park, choi, kwon])

# 코사인 유사도 계산
cos_sim = cosine_similarity(users)

# Kim과 다른 사용자 간의 유사도 추출
kim_similarities = cos_sim[0, 1:]  # Kim (첫 번째 벡터) 제외한 나머지와의 유사도

# 추천: 유사도가 높은 순서대로 정렬
recommendations = np.argsort(-kim_similarities)  # 내림차순으로 정렬

# 사용자 이름 매핑
user_names = ['Lee', 'Park', 'Choi', 'Kwon']
recommended_users = [user_names[i] for i in recommendations]

# 결과 출력
for i, user in enumerate(recommended_users):
    print(f"{i+1}. {user} (유사도: {kim_similarities[recommendations[i]]:.2f})")







