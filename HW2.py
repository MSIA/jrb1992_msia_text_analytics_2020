import nltk
from nltk.tokenize import word_tokenize
import os
import glob
import pandas as pd
import re
import string
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

def preprocess(text):
    # The preprocessing includes tokenization and normalization.
    # Output the preprocessing results as a text file, each line containing a single document, with normalized tokens separated by a single white space.

    # 1. Tokenization
    tokenized = []
    for t in text:
        tokenized.append(nltk.word_tokenize(t))

    # 2. Normalization
    normalized = []
    separated = []
    for word in tokenized:
        # Convert to lower cases and remove non-alphabetic chars
        alphabetic_lower = [i.lower() for i in word if i.isalpha()]
        normalized.append(alphabetic_lower)
        # Normalizaed tokens separated by a single white space
        separated.append(' '.join(alphabetic_lower))

    # 3. Output the preprocessing results as a text file
    with open('Processed_Output.txt', 'w') as f:
        for line in separated:
            # Each line containing a single document
            f.write('%s\n' % line)

    return normalized
    
    
def find_closest_neighbors(model, word_vec):
    # Find 5 closest neighbors for list of words under given models
    ls = []
    for word in word_vec:
        neighbors5 = model.wv.most_similar(word)[:5]
        ls.append(neighbors5)
    return ls


if __name__ == '__main__':

   # Import text corpus
   filepath = os.path.abspath(os.getcwd()) + '/20_newsgroups/alt.atheism/*'
   files = []
   # .glob() retrieves the list of files matching the specified pattern in the file_pattern parameter
   for i in glob.glob(filepath):
       with open(i, 'r', encoding = "utf8", errors = "ignore") as f:
           data = f.read()
           files.append(data)
           f.close()
           
   # Preprocess the text
   normalized = preprocess(files)
   
   # Compare word2vec models with different parameters
   # cbow
   model_cbow1 = Word2Vec(normalized, size = 50, window=5,  workers=4, sg=0)
   model_cbow2 = Word2Vec(normalized, size = 50, window=50, workers=4, sg=0)
   model_cbow3 = Word2Vec(normalized, size = 200, window=50, workers=4, sg=0)
   # skip gram
   model_skipgram = Word2Vec(normalized, size = 50, window=50, workers=4, sg=1)
   # list of models
   models = [model_cbow1, model_cbow2, model_cbow3, model_skipgram]
   models_name = ['model_cbow1', 'model_cbow2', 'model_cbow3', 'model_skipgram']
   
   # Hand-pick 10 words and find 5 closest neighbors for these words under given models
   word_vector = ['central', 'good', 'bad', 'fact', 'evolve', 'large', 'political', 'god', 'science', 'assumptions']

   # Find 5 closest neighbors for list of words under given models and save results to csv
   for i in range(len(models)):
       neighbors = find_closest_neighbors(models[i], word_vector)
       res_df = pd.DataFrame(neighbors, index = word_vector, columns = ['1st', '2nd', '3rd', '4th', '5th'])
       res_df.to_csv(models_name[i] + '.csv')
