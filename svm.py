import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC
import pickle


# Split up train and test datasets
def split_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['binary_label'], test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


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
    
    # 1gram, C=0.5, loss=hinge
    tfidf_1gram = TfidfVectorizer(min_df = 100, max_features = 300, ngram_range=(1,1), sublinear_tf = True, stop_words = 'english')
    train_x_1gram, test_x_1gram = tfidf_transform(tfidf_1gram, X_train, X_test)

    print("Results for 1gram with C=0.5 and loss=hinge")
    svm1 = LinearSVC(C=0.5, loss = 'hinge', random_state=42)
    model1 = svm1.fit(train_x_1gram, y_train)
    pred1 = model1.predict(test_x_1gram)
    score_metrics(y_test, pred1)
    
    # 1gram, C=0.5, loss=squared_hinge
    print("Results for 1gram with C=0.5 and loss=squared_hinge")
    svm2 = LinearSVC(C=0.5, loss = 'squared_hinge', random_state=42)
    model2 = svm2.fit(train_x_1gram, y_train)
    pred2 = model2.predict(test_x_1gram)
    score_metrics(y_test, pred2)
    
    # 1gram+2gram, C=0.5, loss=hinge
    tfidf_1gram2gram = TfidfVectorizer(min_df = 100, max_features = 100, ngram_range=(1,2), sublinear_tf = True, stop_words = 'english')
    train_x_1gram2gram, test_x_1gram2gram = tfidf_transform(tfidf_1gram2gram, X_train, X_test)

    print("Results for 1gram2gram with C=0.5 and loss=hinge")
    svm3 = LinearSVC(C=0.5, loss = 'hinge', random_state=42)
    model3 = svm3.fit(train_x_1gram2gram, y_train)
    pred3 = model3.predict(test_x_1gram2gram)
    score_metrics(y_test, pred3)
    
    # 1gram+2gram, C=5, loss=squared_hinge
    print("Results for 1gram2gram with C=5 and loss=squared_hinge")
    svm4 = LinearSVC(C=5, loss = 'squared_hinge', random_state=42)
    model4 = svm4.fit(train_x_1gram2gram, y_train)
    pred4 = model4.predict(test_x_1gram2gram)
    score_metrics(y_test, pred4)
    
    # Save the best SVM model for Problem 4
    with open('best_model_svm.pkl', 'wb') as f:
       pickle.dump(model3, f)
       
    with open('tfidf_1gram2gram.pkl', 'wb') as f:
       pickle.dump(tfidf_1gram2gram, f)
   
