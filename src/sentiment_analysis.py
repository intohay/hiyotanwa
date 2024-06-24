from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd


df = pd.read_csv('data/hiyoritalk_transcribed.csv')


model = AutoModelForSequenceClassification.from_pretrained('christian-phu/bert-finetuned-japanese-sentiment')
tokenizer = AutoTokenizer.from_pretrained('christian-phu/bert-finetuned-japanese-sentiment', model_max_length=512)

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True)

df.dropna(subset=['text'], inplace=True)


def analyze_sentiment(text):
    result = nlp(text)[0]
    return result



df['sentiment'] = df['text'].apply(analyze_sentiment)
df['sentiment_label'] = df['sentiment'].apply(lambda x: x['label'])
df['sentiment_score'] = df['sentiment'].apply(lambda x: x['score'])


# drop sentiment field
df.drop(columns=['sentiment'], inplace=True)

df.to_csv('data/hiyoritalk_transcribed_sentiment.csv', index=False)

