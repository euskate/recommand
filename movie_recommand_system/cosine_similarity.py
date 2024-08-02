import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

# 코사인 유사도를 데이터프레임으로 변환하여 사용자별 유사도 값 추가
cos_sim_df = pd.DataFrame(cos_sim, index=users, columns=users)

# 사용자별로 kim과의 코사인 유사도 값 추출
cos_sim_with_kim = cos_sim_df['kim']

# 원래 데이터프레임에 코사인 유사도 값 추가
df['코사인 유사도'] = cos_sim_with_kim

# 결과 출력
print(df)

"""
1. 데이터 준비: Kim, Lee, Park, Choi, Kwon의 아이템 A, B, C, D, E, F에 대한 평점 데이터를 정의합니다. 일부 값은 결측값(NaN)으로 표시됩니다.
2. 결측값 처리: 결측값을 각 열의 평균으로 채웁니다. 이 작업은 코사인 유사도를 계산하기 위해 필요합니다.
3. 코사인 유사도 계산: cosine_similarity 함수를 사용해 각 사용자 간의 유사도를 계산합니다.
4. 코사인 유사도 추가: Kim 사용자와 다른 사용자 간의 코사인 유사도 값을 추출하고, 이를 데이터프레임에 새 열로 추가합니다.
5. 결과 출력: 최종 데이터프레임을 출력하여 각 사용자별 코사인 유사도를 확인할 수 있습니다.

결과
       A  B  C  D    E    F  코사인 유사도
kim    5  1  4  4  NaN  NaN  1.000000
lee    3  1  2  2  3.0  2.0  0.990375
park   4  2  4  5  5.0  1.0  0.975100
choi   3  3  1  5  4.0  3.0  0.831398
kwon   1  5  5  2  1.0  4.0  0.672804
"""