import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['label', 'text']

df['label'] = df['label'].replace({0: 0, 4: 1})

# Reduce dataset for faster training
df = df.sample(1000000, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_tf, y_train)

print("Training Accuracy:", model.score(X_train_tf, y_train))
print("Test Accuracy:", model.score(X_test_tf, y_test))

pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("sentiment_vectorizer.pkl", "wb"))

print("Model Saved Successfully!")
