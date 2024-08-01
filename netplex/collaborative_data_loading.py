import pandas as pd

# 파일 경로 설정
file_path = 'u.data'

# 파일을 읽어오고 데이터프레임으로 변환
df = pd.read_csv(file_path, sep='\t', header=None, encoding='utf-8')

# 컬럼명 지정
df.columns = ['UserID', 'ItemID', 'Rating', 'Timestamp']

# 결과 출력
print(df)
