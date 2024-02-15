Subtheme Sentiment Analysis Documentation

Introduction:
Subtheme sentiment analysis involves identifying sentiments associated with specific aspects or problems mentioned in textual data. The task is crucial for understanding customer
feedback and extracting actionable insights from it. In this documentation, we'll outline an approach to subtheme sentiment analysis and demonstrate its implementation using Python code.

Approach:
Our approach consists of several key steps:

Data Preprocessing:

Special characters are removed from the text data, and all text is converted to lowercase. This ensures uniformity and simplifies subsequent processing steps.
The text is tokenized into individual words using the NLTK library's word tokenizer.
Stemming is performed on the tokenized words using the Porter Stemmer algorithm. This reduces words to their root form, aiding in text normalization.
English stopwords, such as "the," "is," and "and," are removed from the tokenized text. Stopwords are common words that do not carry significant meaning and can be safely disregarded.
Sentiment Analysis:

Sentiment analysis is conducted using the VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon, which is specifically designed for sentiment analysis of social media text.
Each text sample is assigned sentiment scores (positive, negative, neutral, and compound) using the VADER lexicon.
Based on the compound score obtained from VADER, sentiment labels (positive, negative, or neutral) are assigned to each text sample.
Machine Learning Model Training:

The preprocessed text data is split into training and testing sets.
A logistic regression classifier is trained on the TF-IDF (Term Frequency-Inverse Document Frequency) vectorized text data. TF-IDF is a numerical statistic that reflects the importance of
a word in a document relative to a collection of documents.
Additionally, a random forest classifier is trained on the TF-IDF vectorized data to explore the performance of a different machine learning algorithm.
Evaluation:

Evaluation metrics such as accuracy, precision, recall, and F1-score are computed to assess the performance of the trained models.
Cross-validation is performed on the logistic regression model to evaluate its generalization performance.
Result:
The implementation of the approach yields insightful results in subtheme sentiment analysis. The trained models demonstrate promising performance in classifying sentiments associated with
different subthemes mentioned in the text data.

Code:
The provided Python code demonstrates the implementation of the approach described above. It includes data preprocessing, sentiment analysis using VADER, training of machine learning
models, and evaluation of model performance.

Summary:
In summary, our approach to subtheme sentiment analysis involves preprocessing text data, conducting sentiment analysis using VADER, training machine learning models, and evaluating
model performance. The implementation demonstrates effective extraction of sentiments associated with specific aspects or problems mentioned in textual data, thereby facilitating better 
understanding of customer feedback.
