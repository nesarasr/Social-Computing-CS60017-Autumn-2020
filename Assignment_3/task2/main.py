'''
Submitted by :

Name : Nesara S R

Roll No : 18IM30014
'''

import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import preprocessor as p
import pathlib
import os
import nltk
import gensim
import multiprocessing
import re
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC





code_path = str(pathlib.Path().absolute())
root_path = code_path[:-6]
pred_path = root_path+'/predictions'
try:
  os.mkdir(pred_path)
except:
  pass


train_path = root_path+'/data/train.tsv'
test_path = root_path+'/data/test.tsv'



'''
Task 2
'''

file1 = open(train_path, 'r')
Lines = file1.readlines()

train_id = []
train_tweet = []
train_label = []

# Read from .tsv files

for line in Lines:
    line = line.replace('\n','')
    values = line.split("\t")
    train_id.append(values[0])
    train_tweet.append(values[1])
    train_label.append(values[2])
    

for i in range(len(train_tweet)):
    train_tweet[i] = ' '.join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", train_tweet[i]).split())

train = pd.DataFrame(list(zip(train_id[1:], train_tweet[1:],train_label[1:])), columns =['id', 'text','hateful'])


file1 = open(test_path, 'r')
Lines = file1.readlines()

test_id = []
test_tweet = []

for line in Lines:
    line = line.replace('\n','')
    values = line.split("\t")
    test_id.append(values[0])
    test_tweet.append(values[1])
    
for i in range(len(test_tweet)):
    test_tweet[i] = ' '.join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", test_tweet[i]).split())

test = pd.DataFrame(list(zip(test_id[1:], test_tweet[1:])), columns =['id', 'text'])



X_train = train['text']
y_train = train['hateful']
X_test = test['text']


stop_re = '\\b'+'\\b|\\b'.join(nltk.corpus.stopwords.words('english'))+'\\b'
X_train = list(X_train.str.replace(stop_re, ''))
X_test = list(X_test.str.replace(stop_re, ''))


train_sentences = []
test_sentences = []
for sentence in X_train:
    train_sentences.append(str(sentence).split())
for sentence in X_test:
    test_sentences.append(str(sentence).split())
    
 
# train word embeddings on the dataset
 
n_workers = multiprocessing.cpu_count()

from gensim.models import Word2Vec
sentences = train_sentences
model = Word2Vec(sentences,size = 300, min_count=1,workers=4)
words = list(model.wv.vocab)

word2vec_size = model[train_sentences[0][0]].shape

def mean_word2vec(sentence):
    feature = []
    for word in sentence:
        feature.append(new_model[word])
    x = np.mean(np.array(feature),axis=0)
    return x
    
def create_adaboost_dataset(train_sentences,test_sentences,y_train,word2vec_size):
    X_train = []
    X_test = []
    for sentence in train_sentences:
        if len(sentence)==0:
            X_train.append(word2vec_size[0]*[0])
        else:
            X_train.append(mean_word2vec(sentence))
    for sentence in test_sentences:
        try:
            if len(sentence)==0:
                X_test.append(word2vec_size[0]*[0])
            else:
                X_test.append(mean_word2vec(sentence))
        except:
            X_test.append(word2vec_size[0]*[0])
    return np.array(X_train),np.array(X_test),np.array(y_train)
    
X_train, X_test, y_train = create_adaboost_dataset(train_sentences,test_sentences,y_train,word2vec_size)

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(base_estimator = RandomForestClassifier(max_depth=8),n_estimators=8)
clf.fit(X_train, y_train)  # Train adaboost classifier with Random Forest as the base estimator



os.chdir(pred_path)

y_test_predict = clf.predict(X_test)
T2_path = pred_path+'/T2.tsv'
T2 = pd.DataFrame(list(zip(test_id[1:], y_test_predict)), columns =['id', 'hateful'])
T2.to_csv(T2_path,index=False)
os.chdir(root_path)














