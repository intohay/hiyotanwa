import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv('../data/hiyoritalk.csv')

# published_atフィールドをISO8601形式としてUTCのdatetime型に変換
df['published_at'] = pd.to_datetime(df['published_at'], utc=True, errors='coerce')

# 日本時間に変換
df['published_at'] = df['published_at'] + pd.Timedelta(hours=9)

# 1時間ごとのメッセージ頻度を集計
df['hour'] = df['published_at'].dt.hour  # 時間単位に変換
hourly_counts = df['hour'].value_counts().sort_index()

# 1時間ごとのメッセージ頻度を棒グラフで視覚化
plt.figure(figsize=(12, 6))
plt.bar(hourly_counts.index, hourly_counts.values, width=0.8)
plt.title('Hourly Message Frequency (Japan Time)')
plt.xlabel('Hour')
plt.ylabel('Message Count')
plt.xticks(range(24))
plt.show()

# 各日の最初のメッセージを取得
df['date'] = df['published_at'].dt.date
first_messages = df.groupby('date')['published_at'].min().dt.hour

# 初めてのメッセージの送信時刻の分布を棒グラフで視覚化
first_message_counts = first_messages.value_counts().sort_index()

plt.figure(figsize=(12, 6))
plt.bar(first_message_counts.index, first_message_counts.values, width=0.8)
plt.title('Distribution of First Message Times (Japan Time)')
plt.xlabel('Hour of First Message')
plt.ylabel('Frequency')
plt.xticks(range(24))
plt.show()
