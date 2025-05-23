import pandas as pd
import re
import string
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
df = pd.read_csv('data/classified_disaster_tweets.csv')
print(df.columns)
df.rename(columns={'None': 'label'}, inplace=True)
df['label'] = df['label'].str.lower()
print("Value counts before filtering:\n", df['label'].value_counts())
valid_labels = ['flood', 'cyclone', 'heatwave', 'none']
df = df[df['label'].isin(valid_labels)]
df['label'] = df['label'].map({'flood': 0, 'cyclone': 1, 'heatwave': 2, 'none': 3})
df = df.dropna(subset=['label'])
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
df['clean_text'] = df['text'].apply(clean_text)
X = df['clean_text']
y = df['label']
print("Any NaNs in y?", y.isna().sum())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/disaster_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print("Model training complete and saved.")


