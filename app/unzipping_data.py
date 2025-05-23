import zipfile
import pandas as pd
import os
zip_path = 'data/nlp-getting-started.zip'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('data/')
    print("Unzipped successfully.")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
submission_df = pd.read_csv('data/sample_submission.csv')
print(train_df.head())
print(test_df.head())
train_df.to_csv("disaster_tweets.csv", index=False)
test_df.to_csv("disaster_tweets.csv", index=False)