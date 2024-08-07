import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render

def index_view(request):
    if request.method == 'POST':
        # 사용자로부터 학번 입력 받기
        school_no = int(request.POST.get('school_no'))

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

        # school_no가 입력된 학번인 학생과 다른 학생들의 유사도
        target_school_no = school_no
        target_similarities = cosine_sim_df.loc[target_school_no].drop(target_school_no)

        # 가장 유사한 학생 선택
        most_similar_school_no = target_similarities.idxmax()

        # 해당 레코드 필터링
        record1 = test_question[(test_question['school_no'] == most_similar_school_no) & (test_question['Chapter'] == 'Chapter5')]
        record2 = test_question[(test_question['school_no'] == most_similar_school_no) & (test_question['Chapter'] == 'Chapter6')]

        # Difficulty 값 불러오기
        difficulty1 = int(record1['Difficulty'].values[0])
        difficulty2 = int(record2['Difficulty'].values[0])

        # test_question에서 랜덤수 불러오기
        random1 = int(test_question[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5')]['Random_number'].values[0])
        random2 = int(test_question[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6')]['Random_number'].values[0])

        # test_question에 난이도 저장
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5'), 'Difficulty'] = difficulty1
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6'), 'Difficulty'] = difficulty2

        # test_bank에서 문제의 레코드 불러오기
        bank1 = test_bank[(test_bank['Chapter'] == 'Chapter5') & (test_bank['Difficulty'] == difficulty1) & (test_bank['Random_number'] == random1)]
        bank2 = test_bank[(test_bank['Chapter'] == 'Chapter6') & (test_bank['Difficulty'] == difficulty2) & (test_bank['Random_number'] == random2)]

        # Problem_number 값 불러오기
        p1 = bank1['Problem_number'].values[0]
        p2 = bank2['Problem_number'].values[0]

        # test_question에 문제번호 저장
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5'), 'Problem_number'] = p1
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6'), 'Problem_number'] = p2

        # 결과 저장
        test_question.to_csv('updated_test_question.csv', index=False)

        # updated_test_question 데이터 로드
        updated_test_question = pd.read_csv('updated_test_question.csv')

        # 문제 번호만 추출
        user_test = updated_test_question[(updated_test_question['school_no'] == school_no)]
        test_loading = user_test[['Problem_number']]

        # test_bank에서 문제 정보 필터링
        test_paper = test_bank[test_bank['Problem_number'].isin(test_loading['Problem_number'])]
        test_paper = test_paper[['Chapter', 'Problem_content', 'Problem_answer']]

        return render(request, 'results.html', {'test_paper': test_paper.to_dict(orient='records')})

    return render(request, 'index.html')

def chat_view(request):
    if request.method == 'POST':
        # 사용자로부터 학번 입력 받기
        school_no = int(request.POST.get('school_no'))

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

        # school_no가 입력된 학번인 학생과 다른 학생들의 유사도
        target_school_no = school_no
        target_similarities = cosine_sim_df.loc[target_school_no].drop(target_school_no)

        # 가장 유사한 학생 선택
        most_similar_school_no = target_similarities.idxmax()

        # 해당 레코드 필터링
        record1 = test_question[(test_question['school_no'] == most_similar_school_no) & (test_question['Chapter'] == 'Chapter5')]
        record2 = test_question[(test_question['school_no'] == most_similar_school_no) & (test_question['Chapter'] == 'Chapter6')]

        # Difficulty 값 불러오기
        difficulty1 = int(record1['Difficulty'].values[0])
        difficulty2 = int(record2['Difficulty'].values[0])

        # test_question에서 랜덤수 불러오기
        random1 = int(test_question[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5')]['Random_number'].values[0])
        random2 = int(test_question[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6')]['Random_number'].values[0])

        # test_question에 난이도 저장
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5'), 'Difficulty'] = difficulty1
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6'), 'Difficulty'] = difficulty2

        # test_bank에서 문제의 레코드 불러오기
        bank1 = test_bank[(test_bank['Chapter'] == 'Chapter5') & (test_bank['Difficulty'] == difficulty1) & (test_bank['Random_number'] == random1)]
        bank2 = test_bank[(test_bank['Chapter'] == 'Chapter6') & (test_bank['Difficulty'] == difficulty2) & (test_bank['Random_number'] == random2)]

        # Problem_number 값 불러오기
        p1 = bank1['Problem_number'].values[0]
        p2 = bank2['Problem_number'].values[0]

        # test_question에 문제번호 저장
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5'), 'Problem_number'] = p1
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6'), 'Problem_number'] = p2

        # 결과 저장
        test_question.to_csv('updated_test_question.csv', index=False)

        # updated_test_question 데이터 로드
        updated_test_question = pd.read_csv('updated_test_question.csv')

        # 문제 번호만 추출
        user_test = updated_test_question[(updated_test_question['school_no'] == school_no)]
        test_loading = user_test[['Problem_number']]

        # test_bank에서 문제 정보 필터링
        test_paper = test_bank[test_bank['Problem_number'].isin(test_loading['Problem_number'])]
        test_paper = test_paper[['Chapter', 'Problem_content', 'Problem_answer']]

        return render(request, 'results.html', {'test_paper': test_paper.to_dict(orient='records')})

    return render(request, 'index.html')

def recommendation_view(request):
    if request.method == 'POST':
        school_no = int(request.POST.get('school_no'))

        # 데이터 로드
        test_scores = pd.read_csv('../test_bank/test_scores.csv')
        test_question = pd.read_csv('../test_bank/test_question.csv')
        test_bank = pd.read_csv('../test_bank/test_bank.csv')

        # school_no와 Chapter별로 평균 점수 계산
        score_summary = test_scores.groupby(['school_no', 'Chapter']).mean().reset_index()

        # 피벗 테이블 생성 (school_no를 행으로, Chapter를 열로)
        score_pivot = score_summary.pivot(index='school_no', columns='Chapter', values='Point').fillna(0)

        # 코사인 유사도 계산
        cosine_sim = cosine_similarity(score_pivot)

        # 결과를 DataFrame으로 변환
        cosine_sim_df = pd.DataFrame(cosine_sim, index=score_pivot.index, columns=score_pivot.index)

        # school_no가 "1"인 학생과 다른 학생들의 유사도
        target_school_no = school_no
        target_similarities = cosine_sim_df.loc[target_school_no].drop(target_school_no)

        # 가장 유사한 학생 선택
        most_similar_school_no = target_similarities.idxmax()

        # 해당 레코드 필터링
        record1 = test_question[(test_question['school_no'] == most_similar_school_no) & (test_question['Chapter'] == 'Chapter5')]
        record2 = test_question[(test_question['school_no'] == most_similar_school_no) & (test_question['Chapter'] == 'Chapter6')]

        # Difficulty 값 불러오기
        difficulty1 = int(record1['Difficulty'].values[0])
        difficulty2 = int(record2['Difficulty'].values[0])

        # test_question에서 랜덤수 불러오기
        random1 = int(test_question[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5')]['Random_number'].values[0])
        random2 = int(test_question[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6')]['Random_number'].values[0])

        # test_question에 난이도 저장
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5'), 'Difficulty'] = difficulty1
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6'), 'Difficulty'] = difficulty2

        # test_bank에서 문제의 레코드 불러오기
        bank1 = test_bank[(test_bank['Chapter'] == 'Chapter5') & (test_bank['Difficulty'] == difficulty1) & (test_bank['Random_number'] == random1)]
        bank2 = test_bank[(test_bank['Chapter'] == 'Chapter6') & (test_bank['Difficulty'] == difficulty2) & (test_bank['Random_number'] == random2)]

        # Problem_number 값 불러오기
        p1 = bank1['Problem_number'].values[0]
        p2 = bank2['Problem_number'].values[0]

        # test_question에 문제번호 저장
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter5'), 'Problem_number'] = p1
        test_question.loc[(test_question['school_no'] == target_school_no) & (test_question['Chapter'] == 'Chapter6'), 'Problem_number'] = p2

        # 결과 저장
        test_question.to_csv('updated_test_question.csv', index=False)

        updated_test_question = pd.read_csv('../test_bank/updated_test_question.csv')
        test_bank = pd.read_csv('../test_bank/test_bank.csv')

        user_test = updated_test_question[(updated_test_question['school_no'] == school_no)]

        # 문제 번호만 추출
        test_loading = user_test[['Problem_number']]

        # test_bank에서 test_loading의 Problem_number에 해당하는 문제 정보 필터링
        test_paper = test_bank[test_bank['Problem_number'].isin(test_loading['Problem_number'])]

        # 필요한 열만 선택
        test_paper = test_paper[['Chapter', 'Problem_content', 'Problem_answer']]

        # 결과를 템플릿에 전달
        return render(request, 'test_paper.html', {'test_paper': test_paper.to_dict(orient='records')})

    return render(request, 'index.html')
