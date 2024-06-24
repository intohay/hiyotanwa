
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd 
from scipy.stats import mannwhitneyu, levene, pearsonr
import matplotlib.pyplot as plt
import numpy as np
import re
plt.rcParams['font.family'] = 'Hiragino Sans'


# 
def count_exclamations_and_bars(text):
    pass
    
# 「やっほ」「やほ」の後の文字数を数える
def count_char_after_yaho(text):
    pattern = r'^(やほ|やっほ)(.*?)(！|)$'

    match = re.match(pattern, text)
    if match:
        extracted_text = match.group(2) + match.group(3)
        return len(extracted_text)
    
    return 0

def parse_date(date_str):
    return pd.to_datetime(date_str, format='%Y-%m%d-%H%M%S')

    

# ラベル情報をスコアに組み込む
def incorporate_label_into_score(label, score):
    if label == 'positive':
        return score
    elif label == 'negative':
        return -score
    else:
        return 0
    



# テストデータを読み込む
print('Loading data...')
df = pd.read_csv('data/hiyoritalk_transcribed_sentiment.csv')


# 日付をパース
df['date'] = df['date'].apply(parse_date)

# 「やほ」「やっほ」が含まれているかどうかを判定
df['contains_yaho'] = df['text'].apply(lambda x: 'やほ' in x or 'やっほ' in x)
# ただしfilenameが.m4aか.mp4で終わる場合はcontains_yahoをFalseにする
df['filename'] = df['filename'].fillna('')
df['filename_video_audio'] = df['filename'].apply(lambda x: x.endswith('.m4a') or x.endswith('.mp4'))
df['contains_yaho'] = df.apply(lambda x: False if x['filename_video_audio'] else x['contains_yaho'], axis=1)
#「やっほ〜」が含まれていた場合はFalse、それ以外はそのまま
df['contains_yaho'] = df.apply(lambda x: False if 'やっほ〜' in x['text'] else x['contains_yaho'], axis=1)





# 「！」と「ー」の数を数える
# df['exclamations'], df['bars'] = zip(*df['text'].apply(count_char_after_yaho))
df['char_after_yaho'] = df['text'].apply(count_char_after_yaho)

# filenameが.jpgか.m4aか.mp4で終わる場合の感情スコアを1.0に設定
df['filename'] = df['filename'].fillna('')
df['filename_media'] = df['filename'].apply(lambda x: x.endswith('.jpg'))
df['sentiment_label'], df['sentiment_score'] = zip(*df.apply(lambda x: ('positive', 1.0) if x['filename_media'] else (x['sentiment_label'], x['sentiment_score']), axis=1))

# save to csv
df.to_csv('data/hiyoritalk_transcribed_sentiment_adjusted.csv', index=False)

print('Extracting relevant data...')
# 「やほ」文が発話された後1時間の間の文章だけを対象に感情スコアを平均する
yaho_rows = df[df['contains_yaho']]

# リストを保持するための辞書を作成
data_dict = {'date': [], 'avg_adjusted_sentiment_score': []}

# 「やほ」文が発話された後1時間以内の文章を抽出
for index, row in yaho_rows.iterrows():
    yaho_time = row['date']
    end_time = yaho_time + pd.Timedelta(hours=1)
    mask = (df['date'] > yaho_time) & (df['date'] <= end_time) & (~df['contains_yaho'])
    relevant_rows = df[mask]
    if not relevant_rows.empty:
        relevant_scores = relevant_rows.apply(lambda x: incorporate_label_into_score(x['sentiment_label'], x['sentiment_score']), axis=1)
        avg_score = relevant_scores.mean()
        data_dict['date'].append(yaho_time)
        data_dict['avg_adjusted_sentiment_score'].append(avg_score)

# DataFrameに変換
avg_sentiments = pd.DataFrame(data_dict)
avg_sentiments.columns = ['date', 'avg_adjusted_sentiment_score']

avg_sentiments['date'] = avg_sentiments['date'].dt.date
print(avg_sentiments.head())


# 「やほ」文の日の「！」と「ー」の合計を計算
char_after_yaho_df = df[df['contains_yaho']].groupby(yaho_rows['date'].dt.date).agg({'char_after_yaho': 'sum'}).reset_index()
char_after_yaho_df.columns = ['date', 'total_char_after_yaho']


# 平均感情スコアとやほ後文の文字数をマージ
merged_df = pd.merge(avg_sentiments, char_after_yaho_df, on='date', how='inner')



# 「！」と「ー」が0個のときとそれ以上のときで感情スコアに差があるかを検定
group_0 = merged_df[merged_df['total_char_after_yaho'] < 3]['avg_adjusted_sentiment_score']
group_1 = merged_df[merged_df['total_char_after_yaho'] >= 3]['avg_adjusted_sentiment_score']

# 平均値を表示
print(f"Group 0 Mean: {group_0.mean()}")
print(f"Group 1 Mean: {group_1.mean()}")
# 個数を表示
print(f"Group 0 Count: {group_0.count()}")
print(f"Group 1 Count: {group_1.count()}")



# Leveneの検定を実行
levene_stat, levene_p_value = levene(group_0, group_1)

print(f"Levene's test statistic: {levene_stat}")
print(f"Levene's test P-value: {levene_p_value}")

# ピアソンの相関係数を計算

correlation, p_value = pearsonr(merged_df['total_char_after_yaho'], merged_df['avg_adjusted_sentiment_score'])

print(f"Pearson's correlation coefficient: {correlation}")
print(f"P-value: {p_value}")


# Mann-Whitney U検定
u_stat, p_value = mannwhitneyu(group_0, group_1, alternative='less')


print(f"U-statistic: {u_stat}")
print(f"P-value: {p_value}")




# ボックスプロットで差を視覚化
plt.figure(figsize=(8, 6))
plt.boxplot([group_0, group_1], labels=['「やほ」「やっほ」の後に続く文字が3字未満', '「やほ」「やっほ」の後に続く文字が3字以上'])
plt.title('平均感情スコアの比較')
plt.ylabel('平均感情スコア')
plt.savefig('boxplot.png')
plt.close()

# 散布図で関係を視覚化
plt.figure(figsize=(8, 6))
jittered_x = merged_df['total_char_after_yaho'] + np.random.normal(0, 0.1, size=merged_df['total_char_after_yaho'].shape)
jittered_y = merged_df['avg_adjusted_sentiment_score'] + np.random.normal(0, 0.05, size=merged_df['avg_adjusted_sentiment_score'].shape)
plt.scatter(jittered_x, jittered_y, color='purple', alpha=0.5)
plt.title('「やほ」「やっほ」の後に続く文字数 vs. その日の感情スコアの平均')
plt.xlabel('「やほ」「やっほ」の後に続く文字数')
plt.ylabel('その日の感情スコアの平均')
plt.savefig('scatterplot.png')
plt.close()