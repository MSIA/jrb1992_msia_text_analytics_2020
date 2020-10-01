import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import spacy
import os
import re
import glob
import time
import multiprocessing as mp

def nltk_tokenize(data):
    # consider stemming and postagging, we choose word tokenization in this case
    tokenized_words = []
    for dat in data:
        words = word_tokenize(dat)
        for word in words:
            tokenized_words.append(word)
    return tokenized_words

def nltk_stemming(data):
    ps = PorterStemmer()
    tokenized_words = nltk_tokenize(data)
    stemmed_words = []
    for w in tokenized_words:
        stemmed_words.append(ps.stem(w))
    return stemmed_words

def nltk_pos_tagging(data):
    tokenized_words = nltk_tokenize(data)
    pos_tagged = nltk.pos_tag(tokenized_words)
    return pos_tagged

def spacy_tokenize(data):
    # consider stemming and postagging, we choose word tokenization in this case
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 5000000
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    tokenized_words = []
    for dat in data:
        doc = nlp(dat)
        words = [token.text for token in doc]
        for word in words:
            tokenized_words.append(word)  
    return tokenized_words

def spacy_lemmatization(data):
    # cannot perform stemming with spacy
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 5000000
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    for dat in data:
        doc = nlp(dat)
        lemmatization = [token.lemma_ for token in doc]
    return lemmatization   

def spacy_pos_tagging(data):
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 5000000
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    for dat in data:
        doc = nlp(dat)
        pos_tagged = [[token.text,token.pos_] for token in doc]
    return pos_tagged   

# implement parallelization
def nltk_parallel(data):
    tokenized_words = nltk_tokenize(data)
    stemmed_words = nltk_stemming(data)
    pos_tagged = nltk_pos_tagging(data)
    return tokenized_words, stemmed_words, pos_tagged

def spacy_parallel(data):
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 5000000
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    tokenized_words = []
    for dat in data:
        doc = nlp(dat)
        words = [token.text for token in doc]
        lemmatization = [token.lemma_ for token in doc]
        pos_tagged = [[token.text,token.pos_] for token in doc]
        for word in words:
            tokenized_words.append(word)  
    return tokenized_words, lemmatization, pos_tagged

# match all emails in text and compile a set of all found email addresses
def extract_emails(text):
    email = re.findall(r'[a-zA-Z0-9+_.-]+\@[a-zA-Z0-9_.-]+\.[a-zA-Z0-9_.-]+', text)
    return email

# match all dates in text and compile a set of all found dates
def extract_dates(text):
    all_dates = []

    # 1 ~ 2 any digit characters followed by month, then 4 any digit characters
    # e.g. 01 January 2020 or 01 Jan 2020
    dates1 = re.findall(r"[\d]{1,2} [JFMAMSOND]\w* [\d]{4}", text)     

    # month followed by 1 ~ 2 any digit characters, then 4 any digit characters
    # e.g. October 28 1992
    dates2 = re.findall(r"[JFMAMSOND]\w* [\d]{1,2} [\d]{4}", combined_files)    

    # e.g. 01/01/2020    
    dates3 = re.findall(r"\b[\d]{1,2}/[\d]{1,2}/[\d]{4}\b", text)     
 
    # e.g. 01-01-2020
    dates4 = re.findall(r"\b[\d]{4}-[\d]{1,2}-[\d]{1,2}\b", text)    
    
    all_dates = dates1+dates2+dates3+dates4
     
    return all_dates


if __name__ == '__main__':
    # import text corpus
    filepath = os.path.abspath(os.getcwd()) + '/20_newsgroups/alt.atheism/*'
    files = []
    # .glob() retrieves the list of files matching the specified pattern in the file_pattern parameter
    for i in glob.glob(filepath):
        with open(i, 'r', encoding = "utf8", errors = "ignore") as f:
            data = f.read()
            files.append(data)
      
    # Problem 1    
            
    # calculate running time 
    start_time = time.time()
    words = nltk_tokenize(files)
    print("NLTK word tokenization for this text corpus takes: ", (time.time() - start_time))
    start_time = time.time()
    stemming = nltk_stemming(files)
    print("NLTK stemming for this text corpus takes: ", (time.time() - start_time))
    start_time = time.time()
    pos_tagging = nltk_pos_tagging(files)
    print("NLTK pos tagging for this text corpus takes: ", (time.time() - start_time))
    

    start_time = time.time()
    words = spacy_tokenize(files)
    print("Spacy word tokenization for this text corpus takes: ", (time.time() - start_time))
    start_time = time.time()
    lemmatization = spacy_lemmatization(files)
    print("Spacy lemmatization for this text corpus takes: ", (time.time() - start_time))
    start_time = time.time()
    pos_tagging = spacy_pos_tagging(files)
    print("Spacy pos tagging for this text corpus takes: ", (time.time() - start_time))
    
    pool = mp.Pool(mp.cpu_count())
    start_time = time.time()
    nltk_parallelization = pool.map(nltk_parallel, files)
    print("NLTK Parallelization for this text corpus takes: ", (time.time() - start_time))
    pool.close()
            
    # comment out this block of code due to extremely long running time
    
    #    pool = mp.Pool(mp.cpu_count())
    #    start_time = time.time()
    #    spacy_parallelization = pool.map(spacy_parallel, files)
    #    print("Spacy Parallelization for this text corpus takes: ", (time.time() - start_time))
    #    pool.close()

   
    # Problem 2
    
    # compile list into a single string to extract emails and dates
    combined_files = ''.join(files)
    emails = extract_emails(combined_files) 
    dates = extract_dates(combined_files)

    # write to files
    with open('emails.txt', 'w') as f:
        for email in emails:
            f.write('%s\n' % email)
    with open('dates.txt', 'w') as f:
        for date in dates:
            f.write('%s\n' % date)
        
        

    
    
    
    





