import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 파일 경로 설정
file_path = 'u.data'

# 파일을 읽어오고 데이터프레임으로 변환
df = pd.read_csv(file_path, sep='\t', header=None, encoding='utf-8')

# 컬럼명 지정
df.columns = ['UserID', 'ItemID', 'Rating', 'Timestamp']

# UserID를 기준으로 피벗 테이블 생성
pivot_table = df.pivot_table(index='UserID', columns='ItemID', values='Rating', fill_value=0)

# 피벗 테이블의 행(row)과 열(column)로 코사인 유사도 계산
cosine_sim_matrix = cosine_similarity(pivot_table.T)

# 코사인 유사도 행렬을 데이터프레임으로 변환
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=pivot_table.columns, columns=pivot_table.columns)

# 결과 출력
print(cosine_sim_df)