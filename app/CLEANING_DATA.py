import pandas as pd
import re
df = pd.read_csv('disaster_tweets.csv')
df = df.dropna()
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text) 
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
df['clean_text'] = df['text'].apply(clean_text)
df.to_csv("disaster_tweets_cleaned.csv", index=False)