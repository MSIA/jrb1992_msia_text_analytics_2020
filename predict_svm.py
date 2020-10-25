import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC
from nltk import word_tokenize
import json
from nltk.corpus import stopwords
import pickle

# Extract stars, text, and word length of documents after applying tokenization
def pre_process(reviews):
    star_ls = []
    text_ls = []
    doc_len_ls = []
    for review in reviews:
        dat = json.loads(review)
        star = dat['stars']
        text = dat['text']
        star_ls.append(star)
        text_ls.append(text)
        # apply nltk word tokenization
        doc_len_ls.append(len(word_tokenize(text)))
    return star_ls, text_ls, doc_len_ls

# Apply nltk word tokenization, convert to lower cases, remove non-alphabetic chars and stopwords
def tokenization(text, stopwords):
    review_tokenized = [i.lower() for i in word_tokenize(text) if i.isalpha() and i not in stopwords]
    return review_tokenized

# Clean reviews and save into CSV file
def cleaning(stars, texts):
    # List of stopwords that do not add much meaning to a sentence
    stop_words = set(stopwords.words('english'))
    texts_tokenized = [tokenization(i, stop_words) for i in texts]
    cleaned_df = pd.DataFrame({'star': stars, 'text': texts_tokenized})
    # Create binary lables: 1 for ratings > 3, 0 for ratings <= 3. This variable is created to apply logistic regression on multi-label class.
    cleaned_df['binary_label'] = cleaned_df.apply(lambda x: 1 if x['star'] > 3 else 0, axis=1)
    # Remove NAs
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

if __name__ == "__main__":

    # import vectorizer and svm model
    with open('tfidf_1gram2gram.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('best_model_svm.pkl', 'rb') as f:
        best_svm_model = pickle.load(f)
    # Read in next 50,000 lines of Yelp reviews for prediction
    lines = open('yelp_academic_dataset_review.json', encoding="utf8").readlines()[600000:650000]
    star_ls, text_ls, doc_len_ls = pre_process(lines)
    cleaned_df = cleaning(star_ls, text_ls)
    
    # Apply tfidf and make predictions
    features = tfidf_vectorizer.fit_transform(text_ls)
    pred = best_svm_model.predict(features)
    conf = best_svm_model.decision_function(features)

    # Save model prediction into json file
    output = {
            # 'review': text_ls, # add this line for pulling reviews
            'actual_label': cleaned_df['binary_label'].tolist(),
            'predicted_label': pred.tolist(),
            'confidence': conf.tolist()
    }
    with open('predcition_results.json', 'w') as f:
        json.dump(output, f)
    print("Prediction results saved into json file.")
