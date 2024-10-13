from openai import OpenAI
from pydantic import BaseModel
import pandas as pd
import umap
import hdbscan
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import glob
from janome.tokenizer import Tokenizer


plt.rcParams['font.family'] = 'Hiragino Sans'
client = OpenAI()

def parse_comments():
   

    prompt = ''''
    「ひよたん」という女性アイドルのインスタグラムのコメントを与えますので、
    そのコメントを「ひよたんの可愛いところ」「ひよたんの好きなところ」「今日食べたもの、あるいは食べる予定のもの」「その他」を分割しなさい。
    「可愛いところ」と「好きなところ」はそれぞれ3つずつ書かれていることが多いことに注意してください。
    ただし一つのコメントに上に上げた4つすべてが記述されているわけではないので、不足している部分は空白としてください。
    インスタの写真(本人が写っている写真)に対する単なる感想は「その他」に分類してください。

    「ひよたんの可愛いところ」は"kawaii"フィールドに、「ひよたんの好きなところ」は"like"フィールドに、「今日食べたもの、あるいは食べる予定のもの」は"food"フィールドに、「その他」は"other"フィールドに記述してください。

    コメント：
    {comment}
    '''


    class Comment(BaseModel):
        kawaii: str
        like: str
        food: str
        other: str


    

    df = pd.read_csv("data/comments.csv")

    for index, row in df.iterrows():
        print(f"{index + 1}/{len(df)}: Processing {row["name"]}'s comment")

        comment = row["comment"]
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt.format(comment=comment)
                }
            ],
            response_format=Comment
        )

        result = completion.choices[0].message.parsed
        
        df.loc[index, "kawaii"] = result.kawaii
        df.loc[index, "like"] = result.like
        df.loc[index, "food"] = result.food
        df.loc[index, "other"] = result.other

        df.to_csv("data/comments_parsed.csv", index=False)


def extract_food():
    
    class Food(BaseModel):
        food: str

    df = pd.read_csv("data/food.csv")

    prompt = '''
        コメントを与えるので、そこに登場する食べ物の名前を抽出しなさい。複数個ある場合はカンマで区切ってください。

        例：「昨日はカレーと焼鳥を食べました。」→「カレー,焼き鳥」

        コメント：{comment}

    '''

    for index, row in df.iterrows():
        print(f"{index + 1}/{len(df)}: Processing {row["name"]}'s comment")

        comment = row["food"]
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt.format(comment=comment)
                }
            ],
            response_format=Food
        )

        result = completion.choices[0].message.parsed
        
        df.loc[index, "food"] = result.food

        df.to_csv("data/food_parsed.csv", index=False)

def extract_kawaii():
    
    class Kawaii(BaseModel):
        kawaii: str

    df = pd.read_csv("data/kawaii.csv")

    prompt = '''
        コメントを与えるので、そこに登場する可愛いところを抽出しなさい。複数個ある場合はカンマで区切ってください。

        例：「可愛いと綺麗が合わさってる所、大人っぽい表情と可愛い表情のギャップが好き、スタイルが良い所が好き！」→「可愛いと綺麗が合わさっているところ,大人っぽい表情と可愛い表情のギャップ,スタイルの良さ」

        コメント：{comment}

    '''

    for index, row in df.iterrows():
        print(f"{index + 1}/{len(df)}: Processing {row["name"]}'s comment")

        comment = row["kawaii"]
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt.format(comment=comment)
                }
            ],
            response_format=Kawaii
        )

        result = completion.choices[0].message.parsed
        
        df.loc[index, "kawaii"] = result.kawaii

        df.to_csv("data/kawaii_parsed.csv", index=False)


def extract_like():
    
    class Like(BaseModel):
        like: str

    df = pd.read_csv("data/like.csv")

    prompt = '''
        コメントを与えるので、そこに登場する好きなところを抽出しなさい。複数個ある場合はカンマで区切ってください。

        例：「綺麗な歌声が好き。みーぱんの前では意外と甘えん坊になるところが好き。何よりも仲間思いでしっかりと熱い思いがあるところが好き！」→「綺麗な歌声,みーぱんの前では意外と甘えん坊になるところ,何よりも仲間思いでしっかりと熱い思いがあるところ」

        コメント：{comment}

    '''

    for index, row in df.iterrows():
        print(f"{index + 1}/{len(df)}: Processing {row["name"]}'s comment")

        comment = row["like"]
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt.format(comment=comment)
                }
            ],
            response_format=Like
        )

        result = completion.choices[0].message.parsed
        
        df.loc[index, "like"] = result.like

        df.to_csv("data/like_parsed.csv", index=False)


def clustering_comments(df, embeddings, output_folder):
    # UMAPで次元削減（ここでは2次元に変換）
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embeddings = umap_model.fit_transform(embeddings)

    # HDBSCANでクラスタリング
    hdbscan_model = hdbscan.HDBSCAN(min_samples=15, min_cluster_size=20)
    clusters = hdbscan_model.fit_predict(embeddings)

    # クラスタのユニークなラベルを取得（-1はノイズ）
    unique_clusters = np.unique(clusters)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_clusters)))

    # 結果の可視化
    plt.figure(figsize=(10, 6))
    for cluster, color in zip(unique_clusters, colors):
        cluster_points = embeddings[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=color, s=50, alpha=0.7, edgecolors='black', 
                    label=f'Cluster {cluster}' if cluster != -1 else 'Noise')
    
    plt.title('UMAP + HDBSCAN Clustering Result')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(loc='best')  # 凡例を追加
    plt.show()

    # クラスタごとにCSVファイルを保存
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    data_df = df.copy()
    # cluster列を追加
    data_df['Cluster'] = clusters


    # 各クラスタごとにデータを分割してCSVに保存
    for cluster_label in set(clusters):
        cluster_data = data_df[data_df['Cluster'] == cluster_label]
        cluster_filename = os.path.join(output_folder, f'cluster_{cluster_label}.csv')
        cluster_data.to_csv(cluster_filename, index=False)

def embed(texts):
    results = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
        encoding_format="float"
    )

    embeddings = [np.array(result.embedding) for result in results.data]
    


    return embeddings

def clustering(csv_filepath, output_folder, type):
    df = pd.read_csv(csv_filepath)

    
    df = df.dropna(subset=[type])

    expanded_df = df.assign(text=df[type].str.split(',')).explode('text').reset_index(drop=True)
    

    expanded_df = expanded_df.dropna(subset=['text'])
    comments = expanded_df["text"].tolist()
    
    # print(f"Embedding {len(comments)} comments...")
    embeddings = embed(comments)


    # # クラスタリングとCSVの保存
    clustering_comments(expanded_df, embeddings, output_folder)

def tokenize_japanese(text):
    tokenizer = Tokenizer()
    tokens = [token.surface for token in tokenizer.tokenize(text) if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']]
    return ' '.join(tokens)

def plot_topics_per_cluster(cluster_folder, top_n_words=5):
    # クラスタごとのファイルを取得 cluster_-1.csvは除去
    cluster_files = glob.glob(f"{cluster_folder}/cluster_*.csv")
    cluster_files = [file for file in cluster_files if 'cluster_-1' not in file]

    n_clusters = len(cluster_files)
    
    # 全体のスコア最大値を求める
    max_score = 0
    all_scores = []

    for cluster_file in cluster_files:
        # クラスタデータの読み込み
        df = pd.read_csv(cluster_file)
        text_data = df['text'].dropna().tolist()
        if text_data:
            tokenized_texts = [tokenize_japanese(text) for text in text_data]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(tokenized_texts)
            scores = np.array(tfidf_matrix.sum(axis=0)).flatten()  # 各単語の合計スコア
            max_score = max(max_score, scores.max())
            all_scores.append(scores)

    fig, axs = plt.subplots(n_clusters, 1, figsize=(10, n_clusters * 3), sharex=True)
    fig.suptitle("Topic Word Scores", fontsize=16)
    
    for i, (cluster_file, scores) in enumerate(zip(cluster_files, all_scores)):
        # クラスタIDとデータ取得
        df = pd.read_csv(cluster_file)
        cluster_id = cluster_file.split('_')[-1].split('.')[0]
        
        # トークナイズとTF-IDFベクトル化
        text_data = df['text'].dropna().tolist()
        tokenized_texts = [tokenize_japanese(text) for text in text_data]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(tokenized_texts)
        
        # 上位の単語を抽出
        terms = vectorizer.get_feature_names_out()
        top_indices = scores.argsort()[-top_n_words:][::-1]
        top_words = [terms[i] for i in top_indices]
        top_scores = scores[top_indices]
        
        # 棒グラフの描画
        axs[i].barh(top_words, top_scores, align='center', color=plt.cm.tab10(i))
        # axs[i].set_title(f'Topic {i} - Cluster {cluster_id}')
        axs[i].invert_yaxis()
        axs[i].set_xlim(0, max_score)  # 全クラスタで同じx軸範囲を使用

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # save the plot
    fig.savefig(f"{cluster_folder}/topic_word_scores.png")



if __name__ == "__main__":
    # データの読み込み
    # clustering('data/kawaii_merged.csv', 'data/kawaii', 'kawaii_parsed')
    # clustering('data/like_merged.csv', 'data/like', 'like_parsed')
    # clustering('data/food_merged.csv', 'data/food', 'food_parsed')

    plot_topics_per_cluster('data/food', top_n_words=5)
    # plot_topics_per_cluster('data/like', top_n_words=5)
    # plot_topics_per_cluster('data/like', n_topics=8, top_n_words=5)