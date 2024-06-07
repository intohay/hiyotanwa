
import os
import pandas as pd
from janome.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'Hiragino Sans'


current_dir = os.path.dirname(__file__)
input_csv: str = os.path.join(current_dir, "../data/hiyoritalk_transcribed_checked.csv")

df: pd.DataFrame = pd.read_csv(input_csv)

illeism_list: list[str] = ["ひよたん", "ひよりさん", "ひよりっち", "ひより"]
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
df['date'] = pd.to_datetime(df['date'], format='%Y-%m%d-%H%M%S')
df['month'] = df['date'].dt.to_period('M')

# 月ごとのilleismとfirstpersonの合計を計算
monthly_counts = df.groupby('month').agg({
    'illeism_count': 'sum',
    'firstperson_count': 'sum'
}).reset_index()


# 月ごとのilleismの割合を計算
monthly_counts['illeism_ratio'] = monthly_counts['illeism_count'] / (monthly_counts['illeism_count'] + monthly_counts['firstperson_count'])


start_period = '2022-01'
end_period = '2024-05'

# 指定期間のデータをフィルタリング
monthly_counts = monthly_counts[(monthly_counts['month'] >= start_period) & (monthly_counts['month'] <= end_period)].reset_index(drop=True)

# 月のラベルをフォーマット
monthly_counts['month_str'] = monthly_counts['month'].dt.strftime('%y年%-m月')

# 最小値を見つける
min_ratio = monthly_counts['illeism_ratio'].min()
min_month_idx = monthly_counts['illeism_ratio'].idxmin()

min_month = monthly_counts.loc[monthly_counts['illeism_ratio'].idxmin(), 'month_str']


# グラフを作成
plt.figure(figsize=(10, 6))
plt.plot(monthly_counts['month_str'], monthly_counts['illeism_ratio'], marker='o', linestyle='-', color='orange', linewidth=6, markersize=16)
plt.title('濱岸ひよりの月ごとの再帰三人称の割合')
plt.xlabel('月')
plt.ylabel('再帰三人称の割合')
plt.grid(True)




plt.annotate(f'{min_ratio:.1%}', xy=(min_month_idx, min_ratio), xytext=(-125, 0),
             textcoords='offset points', fontsize=25, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle="->", color='red', lw=3))

plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

# x軸のラベルを間引いて表示
xticks = monthly_counts['month_str']
plt.xticks(xticks[::3], rotation=45, fontsize=20)  # 3か月ごとに表示

# y軸のラベルを太く表示
plt.yticks(fontsize=20)

plt.tight_layout()

# グラフを表示
plt.show()
