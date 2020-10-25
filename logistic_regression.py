import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression


# Split up train and test datasets
def split_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['binary_label'], test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


# TfidfVectorizer Documentation
# Convert a collection of raw documents to a matrix of TF-IDF features. Equivalent to CountVectorizer followed by TfidfTransformer.
# ngram_range: (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
# min_df: as building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
# sublinear_tfbool: apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
# norm: ‘l2’(Default): Sum of squares of vector elements is 1. The cosine similarity between two vectors is their dot product when l2 norm has been applied. ‘l1’: Sum of absolute values of vector elements is 1.
# max_features: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

# Apply tfidf transformation
def tfidf_transform(vectorizer, train, test):
    features_train = vectorizer.fit_transform(train)
    features_test = vectorizer.fit_transform(test)
    return features_train, features_test
    

# Score metrics for models
def score_metrics(test, pred):
    print("Precision:", precision_score(test, pred))
    print("Recall: ", recall_score(test, pred))
    print("F1 score: ", f1_score(test, pred))
    print("Accuracy: ", accuracy_score(test, pred))
    print("Micro-averaged F1-score: ", f1_score(test, pred, average = 'micro'))


if __name__ == "__main__":

    # Read data in from CSV
    data = pd.read_csv("cleaned_reviews.csv")
    data = data.drop('Unnamed: 0', axis = 1)

    # Split data into test and train sets
    X_train, X_test, y_train, y_test = split_data(data)
    
    # 1gram, C=0.5, max_iter = 400
    tfidf_1gram = TfidfVectorizer(min_df = 100, max_features = 300, ngram_range=(1,1), sublinear_tf = True, stop_words = 'english')
    train_x_1gram, test_x_1gram = tfidf_transform(tfidf_1gram, X_train, X_test)

    print("Results for 1gram with C=0.5 and max_iter = 400")
    lr1 = LogisticRegression(C=0.5, random_state=42, max_iter=400)
    model1 = lr1.fit(train_x_1gram, y_train)
    pred1 = model1.predict(test_x_1gram)
    score_metrics(y_test, pred1)
    
    # 1gram, C=5, max_iter = 400
    print("Results for 1gram with C=5 and max_iter = 400")
    lr2 = LogisticRegression(C=5, random_state=42, max_iter=400)
    model2 = lr2.fit(train_x_1gram, y_train)
    pred2 = model2.predict(test_x_1gram)
    score_metrics(y_test, pred2)
    
    # 1gram+2gram, C=0.5, max_iter = 400
    tfidf_1gram2gram = TfidfVectorizer(min_df = 100, max_features = 100, ngram_range=(1,2), sublinear_tf = True, stop_words = 'english')
    train_x_1gram2gram, test_x_1gram2gram = tfidf_transform(tfidf_1gram2gram, X_train, X_test)

    print("Results for 1gram2gram with C=0.5 and max_iter = 400")
    lr3 = LogisticRegression(C=0.5, random_state=42, max_iter=400)
    model3 = lr3.fit(train_x_1gram2gram, y_train)
    pred3 = model3.predict(test_x_1gram2gram)
    score_metrics(y_test, pred3)

    # 1gram+2gram, C=5, max_iter = 100
    print("Results for 1gram2gram with C=5 and max_iter = 100")
    lr4 = LogisticRegression(C=5, random_state=42, max_iter=100)
    model4 = lr4.fit(train_x_1gram2gram, y_train)
    pred4 = model4.predict(test_x_1gram2gram)
    score_metrics(y_test, pred4)
