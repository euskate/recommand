import numpy as np
from sklearn.metrics import mean_squared_error

# 실제 값과 예측 값
true_values = [4, 3]
pred_values = [3.7, 2.2]

# MSE 계산
mse = mean_squared_error(true_values, pred_values)

# RMSE 계산
rmse = np.sqrt(mse)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


"""
MSE (Mean Squared Error) 및 RMSE (Root Mean Squared Error)를 계산하는 과정은 다음과 같습니다:

예측 값과 실제 값을 준비:

예측 값은 추천 시스템에서 계산된 값입니다.
실제 값은 우리가 수동으로 설정한 값입니다.
MSE 계산:

각 예측 값과 실제 값의 차이를 제곱합니다.
제곱된 차이들의 평균을 구합니다.
RMSE 계산:

MSE의 제곱근을 구합니다.

1. 예측 값과 실제 값 준비
실제 값 (True values): true_values = [4, 3]
예측 값 (Predicted values): pred_values = [3.7, 2.2]


2. MSE 계산
각 예측 값과 실제 값의 차이를 제곱합니다:

(4 - 3.7)^2 = 0.3^2 = 0.09
(3 - 2.2)^2 = 0.8^2 = 0.64

차이의 제곱을 더한 후 평균을 구합니다:

MSE = (0.09 + 0.64) / 2 = 0.735 / 2 = 0.365


3. RMSE 계산
MSE의 제곱근을 구합니다:

RMSE = sqrt(0.365) ≈ 0.6033
"""