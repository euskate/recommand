# KNN(최근접이웃) 에 의한 추천 시스템
import numpy as np
from sklearn.impute import KNNImputer

# 사용자 벡터 정의 (Kim의 다섯 번째, 여섯 번째 값은 결측값으로 처리)
kim = np.array([5, 1, 4, 4, np.nan, np.nan])
lee = np.array([3, 1, 2, 2, 3, 2])
park = np.array([4, 2, 4, 5, 5, 1])
choi = np.array([3, 3, 1, 5, 4, 3])
kwon = np.array([1, 5, 5, 2, 1, 4])

# 모든 사용자 벡터를 모아서 행렬로 만듦
users = np.array([kim, lee, park, choi, kwon])

# KNNImputer를 사용해 결측값 채우기
knn_imputer = KNNImputer(n_neighbors=2, weights='uniform')  # 가까운 2명의 이웃을 사용
users_filled = knn_imputer.fit_transform(users)

# Kim의 채워진 벡터 확인
kim_filled = users_filled[0]

# Kim의 다섯 번째와 여섯 번째 값에 대한 레이블 결정
fifth_value = kim_filled[4]
sixth_value = kim_filled[5]

if fifth_value > sixth_value:
    label = "E"
else:
    label = "F"

# 결과 출력
print(f"Kim의 다섯 번째 값: {fifth_value}")
print(f"Kim의 여섯 번째 값: {sixth_value}")
print(f"추천: {label}")