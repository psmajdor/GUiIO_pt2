import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

df = pd.read_csv("training.1600000.processed.noemoticon.csv",
                 encoding='latin1',
                 header=None,
                 usecols=[0, 5],
                 names=['sentiment', 'text'])

df['sentiment'] = df['sentiment'].replace({4: 1})

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

print(df.head())

X = df['clean_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

train_df = pd.DataFrame({'text': X_train, 'sentiment': y_train})
test_df  = pd.DataFrame({'text': X_test,  'sentiment': y_test})

train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)