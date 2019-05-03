import nltk
from nltk.stem.porter import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
stop = set(stopwords.words("english"))
import warnings
warnings.filterwarnings('ignore')
import os

import pandas as pd
#extract lemma

def dataset(train_path,test_path,valid_path):
    train = pd.read_csv(train_path,delimiter='\t',header=None,encoding='utf8',error_bad_lines=False)
    test = pd.read_csv(test_path,delimiter='\t',header=None,encoding='utf8',error_bad_lines=False)
    valid = pd.read_csv(valid_path,delimiter='\t',header=None,encoding='utf8',error_bad_lines=False)
#     train = train.sample(frac=1)
       
    return train,test,valid

def get_dataset(train_path,test_path):
    train = pd.read_csv(train_path,delimiter='\t',header=None)
    test = pd.read_csv(test_path,delimiter='\t',header=None)
    
#     train = train.sample(frac=1)
    
    x_train = train[2]
    y_train = train[1]
    x_test = test[2]
    y_test = test[1]
    
    return x_train,y_train, x_test, y_test

def get_dataset_art(train_path,test_path):
    train = pd.read_csv(train_path,delimiter='\t',header=None)
    test = pd.read_csv(test_path,delimiter='\t',header=None)
    
#     train = train.sample(frac=1)
    
    x_train = train[14]
    y_train = train[1]
    x_test = test[14]
    y_test = test[1]
    
    return x_train,y_train, x_test, y_test

def load_dataset():
    names = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
            "barely-true", "false", "half-true", "mostly-true", "pants-fire", "venue","article"]
        
    train= pd.read_csv("LIAR-PLUS/dataset/train.tsv",delimiter='\t',header=None,
                       error_bad_lines=False,names=names)
    test= pd.read_csv("LIAR-PLUS/dataset/test.tsv",delimiter='\t',header=None,
                      error_bad_lines=False,names=names)
    valid= pd.read_csv("LIAR-PLUS/dataset/valid.tsv",delimiter='\t',header=None,
                       error_bad_lines=False,names=names)

    return train,test,valid

    #praproses
def text_to_wordlist(text, remove_stopwords=False,stem_words=False):
    
    text = text.lower().split()
    
    #remove stops
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        
    text = " ".join(text)
    
    #clean text 
    text = re.sub(r"'s ", " is ", text)
    text = re.sub(r"’s ", " is ", text)
    #text = re.sub(r"u.s", "us", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r",", "", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    
    
    # stemmer
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        
    return text

#get Top*N word
def get_resume(df, vec, ascending = False, n = None):
    X = vec.fit_transform(df.values)
    feature_names = vec.get_feature_names()

    resume = pd.DataFrame(columns = feature_names, data = X.toarray()).sum()

    if(n):
        return resume.sort_values(ascending = ascending)[:n]

    return resume



