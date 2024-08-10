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
        return None, None, None, None, None, None, None, '추천된 아이템이 사용자의 평가 목록에 없습니다.'

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


def find_similar_users_for_items(recommendations, pivot_table, cosine_sim_df, num_similar_users=3):
    similar_users_info = {}
    for item in recommendations:
        if item not in cosine_sim_df.index:
            print(f"Item {item}는 데이터에 존재하지 않습니다.")
            continue

        # 해당 아이템과 유사한 사용자를 찾기 위해 아이템의 인덱스 가져오기
        item_index = pivot_table.columns.get_loc(item)
        item_vector = pivot_table.T.iloc[item_index].values.reshape(1, -1)

        # 아이템과 유사한 사용자 유사도 계산
        item_similarities = cosine_similarity(item_vector, pivot_table.T)[0]

        # 유사도 높은 상위 사용자 ID 및 평점 추출
        similar_users = np.argsort(item_similarities)[::-1]
        top_similar_users = [user_id for user_id in similar_users if user_id in pivot_table.index][:num_similar_users]

        # 유사도 높은 사용자 및 평점 정보 저장
        similar_users_info[item] = [(user_id, item_similarities[user_id]) for user_id in top_similar_users]

    return similar_users_info


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
    similar_users_info = find_similar_users_for_items(recommendations, pivot_table, cosine_sim_df)

    if mse is not None:
        print(f"User {user_id} 추천 아이템: {recommendations_titles}")
        print(f"추천 아이템 및 평점: {list(zip(recommendations, avg_item_ratings))}")
        print(f"추천 아이템의 평균 평점: {avg_rating:.2f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"모델 성능: {performance}")

        # 각 추천된 아이템별로 유사도 높은 사용자 및 평점 출력
        print("추천된 아이템별 유사한 사용자 및 평점:")
        for item, users_info in similar_users_info.items():
            print(f"Item {item} ({items_df.loc[item, 'Title']}):")
            for user_id, similarity in users_info:
                print(f"  User {user_id}: {similarity:.4f}")
    else:
        print(avg_item_ratings)  # 오류 메시지 출력
else:
    print(f"User {user_id}는 데이터에 존재하지 않습니다.")
