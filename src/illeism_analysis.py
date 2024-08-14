
import os
import pandas as pd
from janome.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'Hiragino Sans'


current_dir = os.path.dirname(__file__)
input_csv: str = os.path.join(current_dir, "../data/hiyoritalk_transcribed_checked.csv")

df: pd.DataFrame = pd.read_csv(input_csv)

illeism_list: list[str] = ["ひよたん", "ひよりさん", "ひよりっち", "ひよりちゃん", "ひよりママ"]
firstperson_list: list[str] = ["私", "わたし", "あたし"]


tokenizer: Tokenizer = Tokenizer()

def count_illeism(text: str) -> int:
    count: int = 0
    
    for illeism in illeism_list:
        count += text.count(illeism)
        text = text.replace(illeism, "[illeism]")

    return count


def count_firstperson(text: str) -> int:
    count: int = 0

    # 「わたしに行った」のような表現は一人称として数えてしまう
    for token in tokenizer.tokenize(text):
        surface = token.surface
        part_of_speech = token.part_of_speech.split(',')
        if surface in firstperson_list and part_of_speech[0] == "名詞" and part_of_speech[1] == "代名詞":
            count += 1

    return count


# textがnanの行を削除
df = df.dropna(subset=['text'])

# textフィールドに対して関数を適用
df['illeism_count'] = df['text'].apply(count_illeism)
df['firstperson_count'] = df['text'].apply(count_firstperson)


# dateフィールドをdatetime型に変換し、月ごとに集計
df['datetime'] = pd.to_datetime(df['date'], format='%Y-%m%d-%H%M%S')
df['month'] = df['datetime'].dt.to_period('M')




# 月ごとのilleismとfirstpersonの合計を計算
monthly_counts = df.groupby('month').agg({
    'illeism_count': 'sum',
    'firstperson_count': 'sum'
}).reset_index()


# 月ごとのilleismの割合を計算
monthly_counts['illeism_ratio'] = monthly_counts['illeism_count'] / (monthly_counts['illeism_count'] + monthly_counts['firstperson_count'])


# start_period = '2022-01'
# end_period = '2024-05'

# # 指定期間のデータをフィルタリング
# monthly_counts = monthly_counts[(monthly_counts['month'] >= start_period) & (monthly_counts['month'] <= end_period)].reset_index(drop=True)

# 月のラベルをフォーマット
monthly_counts['month_str'] = monthly_counts['month'].dt.strftime('%y年%-m月')

# 最小値を見つける
min_ratio = monthly_counts['illeism_ratio'].min()
min_month_idx = monthly_counts['illeism_ratio'].idxmin()

min_month = monthly_counts.loc[monthly_counts['illeism_ratio'].idxmin(), 'month_str']


# グラフを作成
plt.figure(figsize=(10, 6))
plt.plot(monthly_counts['month_str'], monthly_counts['illeism_ratio'], marker='o', linestyle='-', color='orange', linewidth=2, markersize=5)
plt.title('濱岸ひよりの月ごとの再帰三人称の割合の推移')
plt.xlabel('月')
plt.ylabel('再帰三人称の割合')
plt.grid(True)




plt.annotate(f'{min_ratio:.1%}', xy=(min_month_idx, min_ratio), xytext=(-125, 0),
             textcoords='offset points', fontsize=25, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle="->", color='red', lw=3))

plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

# x軸のラベルを間引いて表示
xticks = monthly_counts['month_str']
plt.xticks(xticks[::3], rotation=45, fontsize=10)  # 3か月ごとに表示

# y軸のラベルを太く表示
plt.yticks(fontsize=10)

plt.tight_layout()

# グラフを表示
plt.show()




import pandas as pd
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.dates as mdates

# datetimeから日付のみを取り出す
df['date'] = df['datetime'].dt.date

daily_counts = df.groupby('date', as_index=False).agg({
    'illeism_count': 'sum',
    'firstperson_count': 'sum'
}).reset_index()


daily_counts['date'] = pd.to_datetime(daily_counts['date'])
# 日付をインデックスに設定
daily_counts.set_index('date', inplace=True)





# 週ごとにデータをリサンプリング（週の始まりは月曜日）
weekly_counts = daily_counts.resample('W-MON', label='left', closed='left').sum()

# インデックスをリセットして 'date' 列として扱う
weekly_counts.reset_index(inplace=True)

# 週毎のilleismの割合を計算
weekly_counts['illeism_ratio'] = weekly_counts['illeism_count'] / (weekly_counts['illeism_count'] + weekly_counts['firstperson_count'])


# # カルマンフィルタの初期化
kf = KalmanFilter(dim_x=1, dim_z=1)
kf.x = np.array([[0.7]])  # 初期確率の推定値 (例: 0.5)
kf.F = np.array([[1]])    # 状態遷移行列
kf.H = np.array([[1]])    # 観測モデル
kf.P = np.array([[1]])    # 推定誤差共分散行列の初期値
kf.R = np.array([[0.1]])  # 観測ノイズの分散
kf.Q = np.array([[0.01]]) # プロセスノイズの分散

# 推定結果を保存するリスト
estimated_probabilities = []
no_observation_days = []  # 観測がなかった日のインデックスを保存


# カルマンフィルタによる逐次推定とベースライン計算
for i, row in weekly_counts.iterrows():
    if row['illeism_count'] + row['firstperson_count'] > 0:  # 観測がある日のみ更新
        weekly_average = row['illeism_ratio']  # その日の「私」の使用割合
        kf.predict()  # 予測ステップを行う
        kf.update(np.array([[weekly_average]]))
        
    else:
        # 観測がない日: 推定値をそのまま保持し、予測ステップをスキップ
        no_observation_days.append(i)
    
    # 現在の推定値を保存
    estimated_probabilities.append(kf.x[0, 0])



# グラフの描画
plt.figure(figsize=(20, 7))  # 横長のグラフに調整


plt.plot(weekly_counts['date'], estimated_probabilities, label="Kalman Filter Estimate", color='darkorange')

# x軸のフォーマットを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y年%-m月'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 月ごとのラベルを設定
plt.gcf().autofmt_xdate(rotation=45, ha='center')  # x軸のラベルを45度傾ける

# y軸に補助線を追加
plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(0.1))
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

plt.xlabel("月")
plt.ylabel("確率")
plt.ylim(0, 1)
plt.title("濱岸ひよりの再帰三人称の推定使用確率の推移")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()