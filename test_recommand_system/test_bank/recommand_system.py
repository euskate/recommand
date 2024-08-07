import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 로드
test_scores = pd.read_csv('test_scores.csv')
test_question = pd.read_csv('test_question.csv')
test_bank = pd.read_csv('test_bank.csv')
test_user = pd.read_csv('test_user.csv')

# school_no와 Chapter별로 평균 점수 계산
score_summary = test_scores.groupby(['school_no', 'Chapter']).mean().reset_index()

# 피벗 테이블 생성 (school_no를 행으로, Chapter를 열로)
score_pivot = score_summary.pivot(index='school_no', columns='Chapter', values='Point').fillna(0)

# 코사인 유사도 계산
cosine_sim = cosine_similarity(score_pivot)

# 결과를 DataFrame으로 변환
cosine_sim_df = pd.DataFrame(cosine_sim, index=score_pivot.index, columns=score_pivot.index)

# school_no가 "1"인 학생과 다른 학생들의 유사도
school_no = int(input("학번을 입력해주시기 바랍니다."))
target_school_no = school_no
target_similarities = cosine_sim_df.loc[target_school_no].drop(target_school_no)

# 가장 유사한 학생 선택
most_similar_school_no = target_similarities.idxmax()

# 유사도가 가장 높은 학생의 ID 출력
print(f"유사도가 가장 높은 학생의 school_no: {most_similar_school_no}")

# 해당 레코드 필터링
record1 = test_question[(test_question['school_no'] == most_similar_school_no) & (test_question['Chapter'] == 'Chapter5')]
record2 = test_question[(test_question['school_no'] == most_similar_school_no) & (test_question['Chapter'] == 'Chapter6')]

print("레코드1 : ", record1)
print("레코드2 : ", record2)

# Difficulty 값 불러오기
difficulty1 = int(record1['Difficulty'].values[0])
print("Chapter5의 난이도", difficulty1)
difficulty2 = int(record2['Difficulty'].values[0])
print("Chapter6의 난이도", difficulty2)

# test_question에서 랜덤수 불러오기
random1 = int(test_question[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5')]['Random_number'].values[0])
random2 = int(test_question[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6')]['Random_number'].values[0])

# test_question에 난이도 저장
test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5'), 'Difficulty'] = difficulty1
test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6'), 'Difficulty'] = difficulty2

print("랜덤수1 : ", random1)
print("랜덤수2 : ", random2)


# test_bank에서 문제의 레코드 불러오기
bank1 = test_bank[(test_bank['Chapter'] == 'Chapter5') & (test_bank['Difficulty'] == difficulty1) & (test_bank['Random_number'] == random1)]
bank2 = test_bank[(test_bank['Chapter'] == 'Chapter6') & (test_bank['Difficulty'] == difficulty2) & (test_bank['Random_number'] == random2)]

# Problem_number 값 불러오기
p1 = bank1['Problem_number'].values[0]
p2 = bank2['Problem_number'].values[0]

print("출제 문제 번호1 : ", p1)
print("출제 문제 번호2 : ", p2)

# test_question에 문제번호 저장
test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5'), 'Problem_number'] = p1
test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6'), 'Problem_number'] = p2

print("난이도와 문제번호 적용\n", test_question)

# 결과 저장
test_question.to_csv('updated_test_question.csv', index=False)

updated_test_question = pd.read_csv('updated_test_question.csv')
test_bank = pd.read_csv('test_bank.csv')

user_test = updated_test_question[(updated_test_question['school_no'] == school_no)]

# 문제 번호만 추출
test_loading = user_test[['Problem_number']]

# test_bank에서 test_loading의 Problem_number에 해당하는 문제 정보 필터링
test_paper = test_bank[test_bank['Problem_number'].isin(test_loading['Problem_number'])]

# 필요한 열만 선택
test_paper = test_paper[['Chapter', 'Problem_content', 'Problem_answer']]

# 결과 출력
print(test_paper)


