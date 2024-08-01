# 데이터를 로딩하여 코사인 유사도 계산
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

# 코사인 유사도 행렬을 데이터프레임으로 변환
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=pivot_table.index, columns=pivot_table.index)

# 결과 출력
print(cosine_sim_df)

