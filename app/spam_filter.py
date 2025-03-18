import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class SpamFilter:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words="english")
        self.model = MultinomialNB()

    def train(self, data_path):
        # Load training data
        data = pd.read_csv(data_path)
        X = self.vectorizer.fit_transform(data["message"])
        y = data["label"]
        self.model.fit(X, y)

    def predict(self, email):
        X_test = self.vectorizer.transform([email])
        return self.model.predict(X_test)[0]
