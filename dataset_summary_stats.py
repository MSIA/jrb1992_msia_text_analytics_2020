import json
import pandas as pd
import numpy as np
import multiprocessing
from nltk.tokenize import word_tokenize
from collections import Counter
from statistics import mean
from nltk.corpus import stopwords


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

    # Each review would include the following components:
    # review_id, user_id, busines_id, stars, useful, funny, cool, text, date
    with open("yelp_academic_dataset_review.json") as f:
        preview = [next(f) for x in range(1)]
    # Inspect structure of each review
    print(preview)
    
    # Read in first 500,000 lines of Yelp reviews due to limited computing power
    lines = open('yelp_academic_dataset_review.json', encoding="utf8").readlines()[:500000]
    # Compile list of labels(stars), text, and word length of documents
    star_ls, text_ls, doc_len_ls = pre_process(lines)
    # Number of documents
    num_doc = len(lines)
    # Number of labels
    num_labels = len(set(star_ls))
    # Label distribution
    distribution = Counter(star_ls)
    # Average / mean word length of documents
    avg_len = mean(doc_len_ls)
    print("Number of documents: ", num_doc)
    print("Number of labels: ", num_labels)
    print("Label distribution: ", distribution)
    print("Average word length of documents: ", avg_len)
    
    # Clean reviews and save into CSV file
    print("Review cleaning in process...")
    cleaned_df = cleaning(star_ls, text_ls)
    print("The first 10 rows of cleaned dataset: \n", cleaned_df.head(10))
    cleaned_df.to_csv('cleaned_reviews.csv')
    print("Cleaned dataframe successfully saved into CSV as 'cleaned_reviews.csv'")
    
    
