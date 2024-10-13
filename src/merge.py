import pandas as pd

# CSVファイルの読み込み
df1 = pd.read_csv('data/food.csv')
df2 = pd.read_csv('data/food_parsed.csv')

# 結合（'like'列を'parsed_like'としてリネーム）
merged_df = pd.merge(df1, df2, on='id', suffixes=('', '_parsed'))

# name_parsed列は不要なので削除
merged_df = merged_df.drop(columns='name_parsed')
# 新しいCSVファイルとして保存
merged_df.to_csv('data/food_merged.csv', index=False)

# 結果の確認
print(merged_df)
