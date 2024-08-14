import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 真の確率の設定 (滑らかに変化する確率を生成)
np.random.seed(42)
days = 100
t = np.linspace(0, 2 * np.pi, days)
true_daily_probabilities = 0.5 + 0.5 * np.sin(t) 

# 各日ごとの一人称の使用回数をシミュレート (0: "僕", 1: "私")
# 各日の観測回数もランダムに決定
daily_observations = [np.random.binomial(1, p, size=np.random.randint(0, 10)) for p in true_daily_probabilities]

# カルマンフィルタの初期化
kf = KalmanFilter(dim_x=1, dim_z=1)
kf.x = np.array([[0.5]])  # 初期確率の推定値 (例: 0.5)
kf.F = np.array([[1]])    # 状態遷移行列
kf.H = np.array([[1]])    # 観測モデル
kf.P = np.array([[1]])    # 推定誤差共分散行列の初期値
kf.R = np.array([[0.1]])  # 観測ノイズの分散
kf.Q = np.array([[0.01]]) # プロセスノイズの分散

# 推定結果を保存するリスト
estimated_probabilities = []
no_observation_days = []  # 観測がなかった日のインデックスを保存

# 単純平均モデルの初期化
overall_mean = np.mean([obs.mean() for obs in daily_observations if len(obs) > 0])

# 過去の観測値の移動平均を保存するリスト
moving_average_probabilities = []
cumulative_sum = 0
count = 0

# カルマンフィルタによる逐次推定とベースライン計算
for i, observations in enumerate(daily_observations):
    if len(observations) > 0:  # 観測がある日のみ更新
        daily_average = np.mean(observations)  # その日の「私」の使用割合
        kf.predict()  # 予測ステップを行う
        kf.update(np.array([[daily_average]]))
        cumulative_sum += daily_average
        count += 1
    else:
        # 観測がない日: 推定値をそのまま保持し、予測ステップをスキップ
        no_observation_days.append(i)
    
    # 現在の推定値を保存
    estimated_probabilities.append(kf.x[0, 0])
    # 単純移動平均の推定値を保存
    moving_average_probabilities.append(cumulative_sum / count if count > 0 else overall_mean)

# 結果のプロット
plt.plot(true_daily_probabilities, label="True Daily Probability of '私'", linestyle='--', color='green')
plt.plot(estimated_probabilities, label="Kalman Filter Estimate", color='blue')
plt.plot([overall_mean] * days, label="Overall Mean Estimate", linestyle='-.', color='purple')
plt.plot(moving_average_probabilities, label="Moving Average Estimate", linestyle=':', color='orange')

# 観測がなかった日にマーカーを追加
plt.scatter(no_observation_days, [estimated_probabilities[i] for i in no_observation_days], 
            color='red', marker='x', label='No Observations')

plt.xlabel("Day")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.title("Daily Estimation of Probability of '私'")
plt.legend()
plt.show()

# 評価: MSEとMAEの計算
mse_kf = mean_squared_error(true_daily_probabilities, estimated_probabilities)
mae_kf = mean_absolute_error(true_daily_probabilities, estimated_probabilities)

mse_mean = mean_squared_error(true_daily_probabilities, [overall_mean] * days)
mae_mean = mean_absolute_error(true_daily_probabilities, [overall_mean] * days)

mse_moving_avg = mean_squared_error(true_daily_probabilities, moving_average_probabilities)
mae_moving_avg = mean_absolute_error(true_daily_probabilities, moving_average_probabilities)

print(f"Kalman Filter MSE: {mse_kf:.4f}, MAE: {mae_kf:.4f}")
print(f"Overall Mean MSE: {mse_mean:.4f}, MAE: {mae_mean:.4f}")
print(f"Moving Average MSE: {mse_moving_avg:.4f}, MAE: {mae_moving_avg:.4f}")

