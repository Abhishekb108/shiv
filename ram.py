import pandas as pd
#df = pd.read_csv("")
df = pd.read_csv(r'C:\Users\ADMIN\Downloads\Evaluation-dataset.csv')

print(df.head())
print(df.columns)
print(df.dtypes)
# Remove special characters and convert text to lowercase
df['text_cleaned'] = df[df.columns[0]].str.replace(r'[^a-zA-Z\s]', '').str.lower()

import nltk
nltk.download('punkt')

# Tokenize text into words
df['tokens'] = df['text_cleaned'].apply(nltk.word_tokenize)
from nltk.stem import PorterStemmer

# Initialize Porter Stemmer
stemmer = PorterStemmer()

# Stem tokens
df['stemmed_tokens'] = df['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])
from nltk.corpus import stopwords
nltk.download('stopwords')

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
df['tokens_filtered'] = df['stemmed_tokens'].apply(lambda x: [word for word in x if word not in stop_words])
# Select final preprocessed data
df_final = df[df.columns[0]]  # Selecting the first column as the final DataFrame
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')  # Download the VADER lexicon
sid = SentimentIntensityAnalyzer()
def analyze_sentiment(text):
    sentiment_score = sid.polarity_scores(text)
    return sentiment_score
df['sentiment_scores'] = df['text_cleaned'].apply(lambda x: analyze_sentiment(x))
df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])
def assign_sentiment_label(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_label'] = df['compound_score'].apply(assign_sentiment_label)
print(df[['text_cleaned', 'compound_score', 'sentiment_label']])
# Import necessary libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Define evaluation metrics
def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text_cleaned'], df['sentiment_label'], test_size=0.2, random_state=42)
# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Create a pipeline with TF-IDF vectorizer and logistic regression classifier
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model
model.fit(X_train, y_train)
# Example: Experiment with a different machine learning algorithm (e.g., Random Forest)
from sklearn.ensemble import RandomForestClassifier

# Create a pipeline with TF-IDF vectorizer and random forest classifier
model_rf = make_pipeline(TfidfVectorizer(), RandomForestClassifier())

# Train the random forest model
model_rf.fit(X_train, y_train)

# Evaluate the performance of the random forest model
y_pred_rf = model_rf.predict(X_test)
evaluate(y_test, y_pred_rf)
# Example: Perform cross-validation to assess generalization performance
from sklearn.model_selection import cross_val_score

# Perform cross-validation on the logistic regression model
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())







