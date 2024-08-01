# 데이터를 로딩하여 출력
import pandas as pd

# 텍스트 파일을 읽어옵니다.
file_path = 'kim.txt'  # 파일 경로 설정

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

# 결과 출력
print(df)