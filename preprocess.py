
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import itertools
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
import spacy
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import string
import re
import nltk
import collections
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from empath import Empath
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import scipy.sparse as sp


def preprocess(text, text_pos, semantics):

    #Tf-idf Bigrams
    #Initialize the `tfidf_vectorizer`
    tfidf_text_vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (2,2), max_features = 20000)
    # Fit and transform the `Text` data
    tfidf_text = tfidf_text_vectorizer.fit_transform([text])
    
    #POS
    #Initialize the `tfidf_vectorizer` 
    tfidf_pos_vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (2,2)) 
    # Fit and transform the training data 
    tfidf_pos = tfidf_pos_vectorizer.fit_transform([text_pos])
    
    #Semantics
    #Initialize the `tfidf_vectorizer` 
    tfidf_sem_vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (1,1)) 
    # Fit and transform the training data 
    tfidf_sem = tfidf_sem_vectorizer.fit_transform([semantics])
    
    #Giving weights to each of the 3 feature vectors generated
    big_w = 0.35 * 3
    synt_w = 0.5 * 3
    sem_w = 0.15 * 3
    
    tfidf_text = big_w*tfidf_text
    tfidf_pos = synt_w*tfidf_pos
    tfidf_sem = sem_w*tfidf_sem
    
    diff_n_rows = tfidf_pos.shape[0] - tfidf_text.shape[0]
    Xb = sp.vstack((tfidf_text, sp.csr_matrix((diff_n_rows, tfidf_text.shape[1])))) 
    #where diff_n_rows is the difference of the number of rows between Xa and Xb
    stack = sp.hstack((tfidf_pos, Xb))
    
    diff_rows = stack.shape[0] - tfidf_sem.shape[0]
    Xb_new = sp.vstack((tfidf_sem, sp.csr_matrix((diff_rows, tfidf_sem.shape[1])))) 
    #where diff_n_rows is the difference of the number of rows between Xa and Xb
    
    return sp.hstack((stack, Xb_new))



