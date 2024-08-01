# 코사인 유사도에 의한 추천 시스템
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 사용자 벡터 정의 (Kim의 다섯 번째, 여섯 번째 값은 0으로 초기화)
kim = np.array([5, 1, 4, 4, 0, 0])
lee = np.array([3, 1, 2, 2, 3, 2])
park = np.array([4, 2, 4, 5, 5, 1])
choi = np.array([3, 3, 1, 5, 4, 3])
kwon = np.array([1, 5, 5, 2, 1, 4])

# 모든 사용자 벡터를 모아서 행렬로 만듦
users = np.array([kim, lee, park, choi, kwon])

# Kim과 다른 사용자 간의 코사인 유사도 계산
cos_sim = cosine_similarity(users)[0, 1:]

# 각 항목의 값을 유사도로 가중 평균하여 예측
def predict_missing_value(kim, others, cos_sim):
    predictions = []
    for i in range(len(kim)):
        if kim[i] == 0:  # 값이 비어 있는 항목에 대해서만 예측
            weighted_sum = np.sum(others[:, i] * cos_sim)
            sum_of_weights = np.sum(cos_sim)
            prediction = weighted_sum / sum_of_weights if sum_of_weights != 0 else 0
            predictions.append(prediction)
        else:
            predictions.append(kim[i])
    return predictions

# 다른 사용자의 벡터 모음 (Kim 제외)
other_users = users[1:]

# Kim의 빈 값을 예측
kim_filled = predict_missing_value(kim, other_users, cos_sim)

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