<div style="font-size:48px;padding:24px;border:2px solid #333;width:800px;text-align:center;margin:50px auto"><h1>영화 추천 시스템 구축 및 평가</h1></div>

<br><br>

------------------------------------------------------------------------------------

<br><br><br><br><br><br><br><br><br>

------------------------------------------------------------------------------------

<div>
    <h1 style="text-align:right">과목명 : AI융합교육평가</h1>
    <h1 style="text-align:right">학&nbsp;&nbsp;&nbsp;&nbsp;번 : 2351034001&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</h1>
    <h1 style="text-align:right">성&nbsp;&nbsp;&nbsp;&nbsp;명 : 김기태&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</h1>
</div>

------------------------------------------------------------------------------------

<br><br><br><br><br><br>


# 목차

- [목차](#목차)
- [소개](#소개)
- [데이터 및 환경 설정](#데이터-및-환경-설정)
  - [데이터 로드 및 전처리](#데이터-로드-및-전처리)
  - [피벗 테이블 생성](#피벗-테이블-생성)
  - [코사인 유사도 계산](#코사인-유사도-계산)
- [추천 시스템 구축](#추천-시스템-구축)
  - [추천 알고리즘 구현](#추천-알고리즘-구현)
  - [유사한 사용자 찾기](#유사한-사용자-찾기)
- [결과 분석](#결과-분석)
  - [추천 아이템](#추천-아이템)
  - [유사한 사용자 분석](#유사한-사용자-분석)
- [전체 코드](#전체-코드)
- [결과 - 사용자 아이디가 2인 경우](#결과---사용자-아이디가-2인-경우)
- [작성 후기](#작성-후기)


<br><br><br><br><br><br><br><br><br><br><br><br>


# 소개

이 보고서에서는 MovieLens 100K 데이터셋을 사용하여 영화 추천 시스템을 구축하고 평가합니다. 시스템의 핵심은 사용자와 아이템 간의 유사도를 기반으로 추천을 생성하는 것입니다. 코사인 유사도와 KNN (K-Nearest Neighbors) 알고리즘을 사용하여 추천 시스템을 구현하였습니다.

<br><br>

# 데이터 및 환경 설정

## 데이터 로드 및 전처리

MovieLens 100K 데이터셋은 영화 추천 시스템을 위한 기본적인 데이터셋으로, 사용자와 영화 간의 평점 정보를 포함하고 있습니다. 데이터는 다음 두 개의 파일로 구성됩니다:

- `u.data`: 사용자 ID, 영화 ID, 평점, 타임스탬프 정보를 포함
- `u.item`: 영화 ID와 제목 정보를 포함

이 데이터셋을 Pandas DataFrame으로 로드하고 필요한 전처리를 진행합니다.

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np

# 파일 경로 설정
file_path = 'u.data'
item_file_path = 'u.item'

# 데이터 파일을 읽어오고 데이터프레임으로 변환
df = pd.read_csv(file_path, sep='\t', header=None, encoding='utf-8')
df.columns = ['UserID', 'ItemID', 'Rating', 'Timestamp']

# 아이템 파일을 읽어오고 데이터프레임으로 변환
items_df = pd.read_csv(item_file_path, sep='|', header=None, encoding='ISO-8859-1', usecols=[0, 1],
                       names=['ItemID', 'Title'], engine='python')
```

<br><br>

## 피벗 테이블 생성

사용자 ID를 기준으로 피벗 테이블을 생성하여 사용자-아이템 매트릭스를 구성합니다.

```python
# UserID를 기준으로 피벗 테이블 생성
pivot_table = df.pivot_table(index='UserID', columns='ItemID', values='Rating', fill_value=0)
```

<br><br>

## 코사인 유사도 계산

피벗 테이블을 기반으로 아이템 간의 코사인 유사도를 계산합니다.

```python
# 피벗 테이블의 행(row)과 열(column)로 코사인 유사도 계산
cosine_sim_matrix = cosine_similarity(pivot_table.T)

# 코사인 유사도 행렬을 데이터프레임으로 변환
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=pivot_table.columns, columns=pivot_table.columns)
```

<br><br><br><br>

# 추천 시스템 구축

## 추천 알고리즘 구현

추천 시스템의 핵심은 사용자에게 맞춤형 추천을 제공하는 것입니다. 이를 위해 코사인 유사도를 사용하여 추천 점수를 계산하고, KNN을 통해 유사한 사용자들을 찾습니다.

```python
def recommend_items(user_id, pivot_table, cosine_sim_df, items_df, num_recommendations=5):
    # 사용자가 평가한 아이템
    user_ratings = pivot_table.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index

    # 사용자 평가에 대한 아이템 유사도 점수 계산
    sim_scores = {}
    for item in unrated_items:
        rated_items = user_ratings[user_ratings > 0].index
        sim_sum = np.sum([cosine_sim_df.loc[item, rated_item] for rated_item in rated_items])
        weighted_sum = np.sum(
            [user_ratings[rated_item] * cosine_sim_df.loc[item, rated_item] for rated_item in rated_items])

        if sim_sum != 0:
            sim_scores[item] = weighted_sum / sim_sum

    # 추천 점수로 정렬
    sorted_items = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [item for item, score in sorted_items[:num_recommendations]]

    # 아이템 제목으로 변환
    recommendations_titles = items_df.set_index('ItemID').loc[recommendations]['Title'].tolist()

    return recommendations, recommendations_titles
```

<br><br>

## 유사한 사용자 찾기

KNN 알고리즘을 사용하여 유사한 사용자들을 찾습니다.

```python
def find_similar_users(user_id, pivot_table, cosine_sim_df, num_similar_users=3):
    # 사용자의 벡터를 가져옵니다
    user_vector = pivot_table.loc[user_id].values.reshape(1, -1)

    # 사용자의 벡터와 모든 다른 사용자 벡터 간의 코사인 유사도 계산
    user_similarities = cosine_similarity(user_vector, pivot_table)[0]

    # 유사도 높은 상위 사용자 ID 추출
    similar_users = np.argsort(user_similarities)[::-1]
    top_similar_users = [(pivot_table.index[user], user_similarities[user]) for user in similar_users if
                         pivot_table.index[user] != user_id][:num_similar_users]

    return top_similar_users
```

<br><br><br><br>

# 결과 분석

## 추천 아이템

추천된 아이템과 그 평점을 출력합니다. 예를 들어, 사용자 ID가 2인 경우:

```python
user_id = 2
recommendations, recommendations_titles, avg_item_ratings, avg_rating, mse, rmse, performance = calculate_metrics(user_id, pivot_table, cosine_sim_df, items_df)
print(f"User {user_id} 추천 아이템: {recommendations_titles}")
print(f"추천 아이템 및 평점: {list(zip(recommendations, avg_item_ratings))}")
print(f"추천 아이템의 평균 평점: {avg_rating:.2f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"모델 성능: {performance}")
```

<br><br>

## 유사한 사용자 분석

- 유사한 사용자와 그들의 코사인 유사도 점수를 출력합니다.

```python
similar_users_info = find_similar_users(user_id, pivot_table, cosine_sim_df)
print("유사한 사용자 및 코사인 유사도:")
for similar_user_id, sim_score in similar_users_info:
    print(f"  User {similar_user_id}: {sim_score:.4f}")
```

<br><br><br><br>


# 전체 코드

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np

# 파일 경로 설정
file_path = 'u.data'
item_file_path = 'u.item'

# 데이터 파일을 읽어오고 데이터프레임으로 변환
df = pd.read_csv(file_path, sep='\t', header=None, encoding='utf-8')
df.columns = ['UserID', 'ItemID', 'Rating', 'Timestamp']

# 아이템 파일을 읽어오고 데이터프레임으로 변환 (인코딩 문제를 해결하기 위해 'ISO-8859-1' 사용)
items_df = pd.read_csv(item_file_path, sep='|', header=None, encoding='ISO-8859-1', usecols=[0, 1],
                       names=['ItemID', 'Title'], engine='python')

# UserID를 기준으로 피벗 테이블 생성
pivot_table = df.pivot_table(index='UserID', columns='ItemID', values='Rating', fill_value=0)

# 피벗 테이블의 행(row)과 열(column)로 코사인 유사도 계산
cosine_sim_matrix = cosine_similarity(pivot_table.T)

# 코사인 유사도 행렬을 데이터프레임으로 변환
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=pivot_table.columns, columns=pivot_table.columns)


def recommend_items(user_id, pivot_table, cosine_sim_df, items_df, num_recommendations=5):
    # 사용자가 평가한 아이템
    user_ratings = pivot_table.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index

    # 사용자 평가에 대한 아이템 유사도 점수 계산
    sim_scores = {}
    for item in unrated_items:
        # 평가된 아이템들에 대한 유사도 점수 계산
        rated_items = user_ratings[user_ratings > 0].index
        sim_sum = np.sum([cosine_sim_df.loc[item, rated_item] for rated_item in rated_items])
        weighted_sum = np.sum(
            [user_ratings[rated_item] * cosine_sim_df.loc[item, rated_item] for rated_item in rated_items])

        if sim_sum != 0:
            sim_scores[item] = weighted_sum / sim_sum

    # 추천 점수로 정렬
    sorted_items = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [item for item, score in sorted_items[:num_recommendations]]

    # 아이템 제목으로 변환
    recommendations_titles = items_df.set_index('ItemID').loc[recommendations]['Title'].tolist()

    return recommendations, recommendations_titles


def calculate_metrics(user_id, pivot_table, cosine_sim_df, items_df, num_recommendations=5):
    # 추천 아이템
    recommendations, recommendations_titles = recommend_items(user_id, pivot_table, cosine_sim_df, items_df,
                                                              num_recommendations)

    # 사용자에게 추천된 아이템과 실제 평점 비교
    user_ratings = pivot_table.loc[user_id]

    # 실제 평점 가져오기
    actual_ratings = user_ratings.loc[recommendations]

    # 추천된 아이템의 평점 가져오기
    recommended_ratings = [user_ratings.loc[item] for item in recommendations]

    if len(recommended_ratings) == 0 or len(actual_ratings) == 0:
        return None, None, None, None, None, None, '추천된 아이템이 사용자의 평가 목록에 없습니다.'

    # MSE와 RMSE 계산
    mse = mean_squared_error(actual_ratings, recommended_ratings)
    rmse = np.sqrt(mse)

    # 추천된 아이템에 대한 평균 평점 계산
    item_ratings = df[df['ItemID'].isin(recommendations)]
    avg_item_ratings = item_ratings.groupby('ItemID')['Rating'].mean().loc[recommendations].values

    # 평균 평점 계산
    avg_rating = np.mean(avg_item_ratings)

    # 정확도 판단
    if rmse < 0.5:
        performance = '매우 우수'
    elif rmse < 1.0:
        performance = '우수'
    elif rmse < 1.5:
        performance = '보통'
    elif rmse < 2.0:
        performance = '저조'
    else:
        performance = '매우 저조'

    return recommendations, recommendations_titles, avg_item_ratings, avg_rating, mse, rmse, performance


def find_similar_users(user_id, pivot_table, cosine_sim_df, num_similar_users=3):
    # 사용자의 벡터를 가져옵니다
    user_vector = pivot_table.loc[user_id].values.reshape(1, -1)

    # 사용자의 벡터와 모든 다른 사용자 벡터 간의 코사인 유사도 계산
    user_similarities = cosine_similarity(user_vector, pivot_table)[0]

    # 유사도 높은 상위 사용자 ID 추출
    similar_users = np.argsort(user_similarities)[::-1]
    top_similar_users = [(pivot_table.index[user], user_similarities[user]) for user in similar_users if
                         pivot_table.index[user] != user_id][:num_similar_users]

    return top_similar_users


# 특정 사용자 ID를 입력받아 추천 아이템을 출력
while True:
    try:
        user_id = int(input("추천을 받을 사용자 ID를 입력하세요 [1-943]: "))
        if user_id < 1 or user_id > 943:
            raise ValueError("ID는 1과 943 사이여야 합니다.")
        break
    except ValueError as ve:
        print(f"잘못된 입력입니다: {ve}")

# 사용자가 피벗 테이블에 존재하는지 확인
if user_id in pivot_table.index:
    recommendations, recommendations_titles, avg_item_ratings, avg_rating, mse, rmse, performance = calculate_metrics(
        user_id, pivot_table, cosine_sim_df, items_df)
    similar_users_info = find_similar_users(user_id, pivot_table, cosine_sim_df)  # 수정된 부분

    if mse is not None:
        print(f"User {user_id} 추천 아이템: {recommendations_titles}")
        print(f"추천 아이템 및 평점: {list(zip(recommendations, avg_item_ratings))}")
        print(f"추천 아이템의 평균 평점: {avg_rating:.2f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"모델 성능: {performance}")

        # 유사한 사용자 및 코사인 유사도 출력
        print("유사한 사용자 및 코사인 유사도:")
        for similar_user_id, sim_score in similar_users_info:
            print(f"  User {similar_user_id}: {sim_score:.4f}")

        # 각 추천된 아이템별로 유사도 높은 사용자 및 평점 출력
        print("추천된 아이템별 유사한 사용자 및 평점:")
        for item, title in zip(recommendations, recommendations_titles):
            print(f"Item {item} ({title}):")
            if item in cosine_sim_df.index:
                similar_users_for_item = cosine_sim_df[item].sort_values(ascending=False).head(3)
                for user_id, sim_score in similar_users_for_item.items():
                    if user_id in pivot_table.index:
                        print(f"  User {user_id}: {sim_score:.4f}")
            else:
                print(f"  Item {item}에 대한 유사도가 계산되지 않았습니다.")
    else:
        print(avg_item_ratings)  # 오류 메시지 출력
else:
    print(f"User {user_id}는 데이터에 존재하지 않습니다.")
```

<br><br><br><br>

# 결과 - 사용자 아이디가 2인 경우

```console
.venv\Scripts\python.exe D:\kim2\python\pythonProject1\netflix_recommand_confirm_real.py 
추천을 받을 사용자 ID를 입력하세요 [1-943]: 2
User 2 추천 아이템: ['They Made Me a Criminal (1939)', 'Death in Brunswick (1991)', 'Mamma Roma (1962)', 'All Things Fair (1996)', 'Shadows (Cienie) (1988)']
추천 아이템 및 평점: [(1122, 5.0), (1593, 4.0), (1674, 4.0), (1619, 3.0), (1546, 1.0)]
추천 아이템의 평균 평점: 3.40
MSE: 0.0000
RMSE: 0.0000
모델 성능: 매우 우수
유사한 사용자 및 코사인 유사도:
  User 701: 0.5806
  User 931: 0.5125
  User 460: 0.5044
추천된 아이템별 유사한 사용자 및 평점:
Item 1122 (They Made Me a Criminal (1939)):
  User 799: 0.5252
Item 1593 (Death in Brunswick (1991)):
Item 1674 (Mamma Roma (1962)):
  User 884: 0.5077
Item 1619 (All Things Fair (1996)):
Item 1546 (Shadows (Cienie) (1988)):
```

<br><br><br><br>

# 작성 후기

이 보고서를 통해 MovieLens 100K 데이터셋을 활용한 영화 추천 시스템의 구축 과정과 결과를 확인할 수 있었습니다. 코사인 유사도와 KNN 알고리즘을 통해 추천 시스템의 정확성을 평가하고, 유사한 사용자 정보를 바탕으로 더 나은 추천을 제공할 수 있음을 알 수 있었습니다. 이에 아울러 그 동안 생각해보지 못했던 코사인 유사도라는 내용을 처음 접하면서 데이터 끼리의 근접도를 계산할 수 있었다는 점은 이 교육 과정을 통해 알게 되었으며, 어떻게 데이터 쌍을 이루어 유사한 지 생각하는 시간이 되었던 것 같습니다.

