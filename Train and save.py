import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import joblib

# Load the data
data = pd.read_excel('immverse_ai_eval_dataset.xlsx') 

# Assuming df is your DataFrame and it includes a 'sentence' column for the text and a 'voice' column for the labels
X = df['sentence']
y = df['voice']

# Vectorize the sentences
vectorizer = CountVectorizer()
X_dtm = vectorizer.fit_transform(X)

# Train the Naive Bayes model
nb = MultinomialNB()
nb.fit(X_dtm, y)

# Save the model and vectorizer
joblib.dump(nb, 'nb_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
