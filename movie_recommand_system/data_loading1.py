import pandas as pd

# 파일 경로 설정 (kim.txt 파일이 있는 위치)
file_path = 'kim.txt'

# 데이터를 읽어오기
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(',')
        data.append(parts)

# 데이터프레임으로 변환
columns = ['User'] + list('ABCDEF')
df = pd.DataFrame(data, columns=['User', 'Item', 'Rating'])
df = df.pivot(index='User', columns='Item', values='Rating').reset_index()

# 결과 출력
print(df)

# 결측값을 NaN으로 변환하고, 정렬
df.replace('', pd.NA, inplace=True)
df = df.reindex(columns=columns)
df.set_index('User', inplace=True)

# DataFrame 출력
print(df)
