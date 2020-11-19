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
Task 1
'''

# READ TRAIN AND TEST SET from ../data/train.tsv and ../data/test.tsv respectively and save them to pandas dataframe

def preprocess_tweet(text):
    text = p.clean(text)
    return text


file1 = open(train_path, 'r')
Lines = file1.readlines()

train_id = []
train_tweet = []
train_label = []


#Read from .tsv files

for line in Lines:
    line = line.replace('\n','')
    values = line.split("\t")
    train_id.append(values[0])
    train_tweet.append(values[1])
    train_label.append(values[2])
    

for i in range(len(train_tweet)):
    train_tweet[i] = preprocess_tweet(train_tweet[i])

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
    test_tweet[i] = preprocess_tweet(test_tweet[i])
    
test = pd.DataFrame(list(zip(test_id[1:], test_tweet[1:])), columns =['id', 'text'])

train.index = train['id']
train = train.drop(['id'],axis=1)

test.index = test['id']
test = test.drop(['id'],axis=1)


#  Task 1(a) : RANDOM FOREST WITH TFIDF VECTORIZATION


X_train = train['text']
y_train = train['hateful']
X_test = test['text']


vectorizer = TfidfVectorizer(lowercase = 'False',max_df=0.8,min_df=5)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

clf = RandomForestClassifier(max_depth=200, random_state=0)
clf.fit(X_train, y_train)    # Train Random Forest Classifier
y_test_predict = list(clf.predict(X_test)) #Predict


os.chdir(pred_path)

rf_path = pred_path+'/RF.tsv'
rf = pd.DataFrame(list(zip(test_id[1:], y_test_predict)), columns =['id', 'hateful'])
rf.to_csv(rf_path,index=False) #Save Result
os.chdir(root_path)



#  Task 1(b) : SVM with Word2Vec

file1 = open(train_path, 'r')
Lines = file1.readlines()

train_id = []
train_tweet = []
train_label = []

#Read from .tsv files


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
    
def create_svm_dataset(train_sentences,test_sentences,y_train,word2vec_size):
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
    
X_train, X_test, y_train = create_svm_dataset(train_sentences,test_sentences,y_train,word2vec_size)

clf = make_pipeline(StandardScaler(), SVC(gamma='scale'))
clf.fit(X_train, y_train)  # Train SVM Classifier

y_test_predict = list(clf.predict(X_test)) #predict


os.chdir(pred_path)

svm_path = pred_path+'/SVM.tsv'
svm = pd.DataFrame(list(zip(test_id[1:], y_test_predict)), columns =['id', 'hateful'])
svm.to_csv(svm_path,index=False) # Save Results
os.chdir(root_path)



    
# Task 1(c) : FastText


import fasttext



file1 = open(train_path, 'r')
Lines = file1.readlines()

train_id = []
train_tweet = []
train_label = []

#Read from .tsv files


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

# Fasttext preprocessing


X_train = train.drop('id',axis=1)
X_test = test.drop('id',axis=1)

X_train,y_train = (X_train['text']),(X_train['hateful'])

s1 = pd.Series(list(y_train), index=list(y_train.index), name='hateful')
s2 = pd.Series(list(X_train), index=list(X_train.index), name='text')
train = pd.concat([s1, s2], axis=1)
train['hateful']=['__label__'+str(s) for s in train['hateful']]
train['text']= train['text'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)
train.to_csv(root_path+'/fasttext_train.txt',index=False,sep=' ', header=False,quoting=csv.QUOTE_NONE,quotechar="", escapechar=" ")

model = fasttext.train_supervised(input=root_path+"/fasttext_train.txt",epoch=25) # Train Model with epoch=25

y_pred = []
for i in range(0,len(test_tweet[1:])):
  x = int(model.predict(test_tweet[1:][i])[0][-1][-1])   #predict
  y_pred.append(x)



ft_path = pred_path+'/FT.tsv'
ft = pd.DataFrame(list(zip(test_id[1:], y_pred)), columns =['id', 'hateful'])
ft.to_csv(ft_path,index=False) # Save results
os.chdir(root_path)















