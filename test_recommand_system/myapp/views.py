import pandas as pd
from django.http import HttpResponseBadRequest
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render, redirect
import os
from django.conf import settings

def load_data():
    test_scores = pd.read_csv(os.path.join(settings.BASE_DIR, 'test_scores.csv'))
    test_question = pd.read_csv(os.path.join(settings.BASE_DIR, 'test_question.csv'))
    test_bank = pd.read_csv(os.path.join(settings.BASE_DIR, 'test_bank.csv'))
    return test_scores, test_question, test_bank

def get_recommended_questions(school_no):
    test_scores, test_question, test_bank = load_data()

    # school_no와 Chapter별로 평균 점수 계산
    score_summary = test_scores.groupby(['school_no', 'Chapter']).mean().reset_index()

    # 피벗 테이블 생성 (school_no를 행으로, Chapter를 열로)
    score_pivot = score_summary.pivot(index='school_no', columns='Chapter', values='Point').fillna(0)

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(score_pivot)

    # 결과를 DataFrame으로 변환
    cosine_sim_df = pd.DataFrame(cosine_sim, index=score_pivot.index, columns=score_pivot.index)

    # 가장 유사한 학생 선택
    target_similarities = cosine_sim_df.loc[school_no].drop(school_no)
    most_similar_school_no = target_similarities.idxmax()

    # 콘솔 출력 추가 - school_no와 코사인 유사도 행렬
    print(f"User's school_no: {school_no}")
    print(f"Cosine Similarity Matrix:\n{cosine_sim_df}\n")
    print(f"Most similar school_no: {most_similar_school_no}")

    # 필요한 레코드 필터링
    record1 = test_question[(test_question['school_no'] == most_similar_school_no) & (test_question['Chapter'] == 'Chapter5')]
    record2 = test_question[(test_question['school_no'] == most_similar_school_no) & (test_question['Chapter'] == 'Chapter6')]

    # 난이도와 랜덤 수를 불러오기
    difficulty1 = int(record1['Difficulty'].values[0])
    difficulty2 = int(record2['Difficulty'].values[0])

    random1 = int(test_question[(test_question['school_no'] == school_no) & (test_question['Chapter'] == 'Chapter5')]['Random_number'].values[0])
    random2 = int(test_question[(test_question['school_no'] == school_no) & (test_question['Chapter'] == 'Chapter6')]['Random_number'].values[0])

    # 콘솔 출력 추가 - Chapter5, Chapter6의 난이도 및 Random_number
    print(f"Chapter5 Difficulty: {difficulty1}, Random_number: {random1}")
    print(f"Chapter6 Difficulty: {difficulty2}, Random_number: {random2}")

    # test_question에 난이도 저장
    test_question.loc[(test_question['school_no'] == school_no) & (test_question['Chapter'] == 'Chapter5'), 'Difficulty'] = difficulty1
    test_question.loc[(test_question['school_no'] == school_no) & (test_question['Chapter'] == 'Chapter6'), 'Difficulty'] = difficulty2

    # test_bank에서 문제의 레코드 불러오기
    bank1 = test_bank[(test_bank['Chapter'] == 'Chapter5') & (test_bank['Difficulty'] == difficulty1) & (test_bank['Random_number'] == random1)]
    bank2 = test_bank[(test_bank['Chapter'] == 'Chapter6') & (test_bank['Difficulty'] == difficulty2) & (test_bank['Random_number'] == random2)]

    # Problem_number 값 불러오기
    p1 = bank1['Problem_number'].values[0]
    p2 = bank2['Problem_number'].values[0]

    # 콘솔 출력 추가 - 추천된 문제번호
    print(f"Assigned Problem_number for Chapter5: {p1}")
    print(f"Assigned Problem_number for Chapter6: {p2}")

    # test_question에 문제번호 저장
    test_question.loc[(test_question['school_no'] == school_no) & (test_question['Chapter'] == 'Chapter5'), 'Problem_number'] = p1
    test_question.loc[(test_question['school_no'] == school_no) & (test_question['Chapter'] == 'Chapter6'), 'Problem_number'] = p2

    # 결과 저장
    test_question.to_csv(os.path.join(settings.BASE_DIR, 'updated_test_question.csv'), index=False)

    # 추천 문제 필터링
    updated_test_question = test_question[test_question['school_no'] == school_no]
    test_loading = updated_test_question[['Problem_number']]
    test_paper = test_bank[test_bank['Problem_number'].isin(test_loading['Problem_number'])]

    # 'is_multiple_choice'와 'options' 컬럼을 선택하지 않고, 존재하는 컬럼만 선택합니다.
    return test_paper[['Problem_number', 'Chapter', 'Problem_content', 'Problem_answer']]



def index_view(request):
    if request.method == 'POST':
        school_no = int(request.POST.get('school_no'))
        test_paper = get_recommended_questions(school_no)
        return render(request, 'results.html', {'test_paper': test_paper.to_dict(orient='records'), 'student_school_no': school_no})

    return render(request, 'index.html')


def submit_answers(request):
    if request.method == 'POST':
        # 폼 데이터에서 학번과 답안을 가져옵니다.
        school_no = request.POST.get('school_no')
        if not school_no:
            return HttpResponseBadRequest("학번이 제공되지 않았습니다.")

        # 폼 데이터에서 문제 번호와 답안을 추출합니다.
        answers = {}
        numbers = request.POST.getlist('problem_numbers')
        problem_numbers = [int(num) for num in numbers]

        chapter_nums = ['Chapter1','Chapter2','Chapter3','Chapter4','Chapter5','Chapter6']
        for chapter_no in chapter_nums:
            answer = request.POST.get(chapter_no)
            if answer:
                answers[chapter_no] = answer.strip()

        # 문제 번호를 정수형으로 변환
        problem_numbers = list(map(int, problem_numbers))

        print(problem_numbers)
        print(answers)

        # CSV 파일 경로 설정
        csv_path = 'updated_test_question.csv'
        test_bank_path = 'test_bank.csv'

        # CSV 파일을 읽어와 DataFrame으로 로드
        test_bank = pd.read_csv(os.path.join(settings.BASE_DIR, 'test_bank.csv'))

        try:
            df = pd.read_csv(csv_path)
            test_bank_df = pd.read_csv(test_bank_path)
        except Exception as e:
            return HttpResponseBadRequest(f"CSV 파일을 로드하는데 문제가 발생했습니다: {e}")

        # 학번이 int형으로 변환 가능한지 확인하고 변환
        try:
            school_no = int(school_no)
        except ValueError:
            return HttpResponseBadRequest("학번은 유효한 숫자여야 합니다.")

        # 문제 번호에 해당하는 데이터만 필터링
        filtered_df = test_bank[test_bank_df['Problem_number'].isin(problem_numbers)]

        # 필요한 컬럼만 선택
        filtered_df = filtered_df[['Problem_number', 'Chapter', 'Problem_answer']]

        # 각 챕터별 점수 매핑
        score_mapping = {
            'Chapter1': 10,
            'Chapter2': 20,
            'Chapter3': 10,
            'Chapter4': 20,
            'Chapter5': 20,
            'Chapter6': 20
        }

        total_score = 0
        grading_results = {}

        # 해당 학번의 행을 찾습니다.
        if school_no in df['school_no'].values:
            print(f"학번: {school_no}")
            num = 0
            for problem_number, answer in answers.items():
                # 문제 번호에 해당하는 정답을 test_bank_df에서 가져옴
                matching_rows = filtered_df[filtered_df['Chapter'] == problem_number]
                if not matching_rows.empty:
                    correct_answer = matching_rows['Problem_answer'].values[0].strip()
                    chapter = matching_rows['Chapter'].values[0]

                    is_correct = answer == correct_answer
                    points = score_mapping.get(chapter, 0) if is_correct else 0
                    total_score += points

                    # 콘솔에 출력
                    print(
                        f"Problem Number: {problem_number}, Submitted Answer: {answer}, Correct Answer: {correct_answer}, Chapter: {chapter}, Score: {points}, Result: {'Correct' if is_correct else 'Incorrect'}")

                    grading_results[problem_number] = {
                        'submitted_answer': answer,
                        'correct_answer': correct_answer,
                        'chapter': chapter,
                        'points': points,
                        'is_correct': is_correct
                    }

                    # 응시자의 답안을 CSV 파일에 저장
                    df.loc[df['school_no'] == school_no, f'Problem_{problem_number}'] = answer
                    df.loc[df['school_no'] == school_no, f'Problem_{problem_number}_Result'] = points
                else:
                    print(f"문제 번호 {problem_number}에 대한 정답이 test_bank.csv에서 찾을 수 없습니다.")
        else:
            new_row = {'school_no': school_no}
            new_row.update(answers)
            df = df.append(new_row, ignore_index=True)

            print(f"학번: {school_no} (신규 응시자)")
            for problem_number, answer in answers.items():
                matching_rows = test_bank_df[test_bank_df['Problem_number'] == problem_number]
                if not matching_rows.empty:
                    correct_answer = matching_rows['Problem_answer'].values[0].strip()
                    chapter = matching_rows['Chapter'].values[0]
                    is_correct = answer == correct_answer
                    points = score_mapping.get(chapter, 0) if is_correct else 0
                    total_score += points

                    # 콘솔에 출력
                    print(
                        f"Problem Number: {problem_number}, Submitted Answer: {answer}, Correct Answer: {correct_answer}, Chapter: {chapter}, Score: {points}, Result: {'Correct' if is_correct else 'Incorrect'}")

                    grading_results[problem_number] = {
                        'submitted_answer': answer,
                        'correct_answer': correct_answer,
                        'chapter': chapter,
                        'points': points,
                        'is_correct': is_correct
                    }
                else:
                    print(f"문제 번호 {problem_number}에 대한 정답이 test_bank.csv에서 찾을 수 없습니다.")

        # 수정된 DataFrame을 다시 CSV 파일로 저장
        df.to_csv(csv_path, index=False)

        # 합격 여부 판단
        pass_status = 'Pass' if total_score >= 60 else 'Fail'

        # 총점과 합격 여부를 콘솔에 출력
        print(f"Total Score: {total_score}, Status: {pass_status}")

        return render(request, 'submission_success.html', {
            'school_no': school_no,
            'grading_results': grading_results,
            'total_score': total_score,
            'pass_status': pass_status
        })

    return HttpResponseBadRequest("잘못된 요청입니다.")


def chat_view(request):
    if request.method == 'POST':
        school_no = int(request.POST.get('school_no'))
        test_paper = get_recommended_questions(school_no)
        return render(request, 'results.html', {'test_paper': test_paper.to_dict(orient='records'), 'student_school_no': school_no})

    return render(request, 'index.html')

def recommendation_view(request):
    if request.method == 'POST':
        school_no = int(request.POST.get('school_no'))
        test_paper = get_recommended_questions(school_no)
        return render(request, 'test_paper.html', {'test_paper': test_paper.to_dict(orient='records'), 'student_school_no': school_no})

    return render(request, 'index.html')

