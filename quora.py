# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 13:14:19 2018

@author: Bolt
"""

"""
road map:
objective:predict duplicate questions
question-words/sentence

step1:once we read the data we need to perform eda
process the data by removing punctuation ,reg expression,letter case,digits
step2:nlp words/entity/relationship/most frequent or important topic
step 3:convert word to vectors/map the vector to collumn an identify the duplicate 
/non duplicate with q1 andq2
step 4:classification models:random forest/gradient boosting/decision tree/xg boost/neural network

"""
import nltk
nltk.download("stopwords")  #it will download the stop words

import numpy as np
import pandas as pd
from collections import Counter  #to identify who many times the word has repeated
import operator
import re
import os
import gc    #same as gensim
import gensim               #topic modeling (identify the most frequent words)
from gensim import corpora   #same as gensim 
from nltk.corpus import stopwords  #i am calling the stop words that i downloaded
import string
from copy import deepcopy


from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer #sparse matrix
from nltk.corpus import stopwords
from nltk import word_tokenize,ngrams
from sklearn import ensemble
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss 


import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
import plotly.offline as py 
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
pal=sns.color_palette()
color=sns.color_palette()
pd.set_option('expand_frame_repr',False)
pd.set_option('display.max_colwidth',-1)

pd.options.mode.chained_assignment=None
words=re.compile(r"\w+",re.I)
stopwords=stopwords.words('english')

#load the dataset
train=pd.read_csv("train.csv").fillna("")  #it will remove na and create a blank space
test=pd.read_csv("test.csv").fillna("")

train.isnull().sum().sum()

for i in range(6):
    print(train.question1[i])
    print(train.question2[i])
    
train.groupby("is_duplicate")["id"].count().plot.bar()

dfs=train[0:2500]
dfs.groupby("is_duplicate")["id"].count().plot.bar()

#seperating qid1 and qid2 into different df
dfq1,dfq2=dfs[['qid1','question1']],dfs[['qid2','question2']]
dfq1.columns=['qid1','question']
dfq2.columns=['qid2','question']

dfqa=pd.concat((dfq1,dfq2),axis=0).fillna("")  #1stcol space question
nrows_for_q1=dfqa.shape[0]/2                   #space 2ndcol question


all_ques_df=pd.DataFrame(pd.concat([train['question1'],train['question2']])) #if we have one column it will be in series so convert it into dataframe
all_ques_df.columns=["questions"]
all_ques_df["num_of_words"]=all_ques_df["questions"].apply(lambda x:len(str(x).split()))

cnt_srs=all_ques_df['num_of_words'].value_counts()
plt.figure(figsize=(12,6))               #to change the size of figuresize
sns.barplot(cnt_srs.index,cnt_srs.values,alpha=0.8,color=color[0])
plt.ylabel('number of occurence',fontsize=12)
plt.xlabel('number of words in question',fontsize=12)
plt.xticks(rotation='vertical')   #rotate the value in xlabel to vertical
plt.show()
'''conclusion=most of the questions are 8 to 10 character long'''

print('Total number of question pairs for training:{}'.format(len(train)))
print('duplicate pairs:{}'.format(round(train['is_duplicate'].mean()*100,2)))
qids=pd.Series(train['qid1'].tolist() + train['qid2'].tolist()) #instead of concat we can use list to merge
print('total number of question in training data:{}'.format(len(np.unique(qids))))
print('total number of question that appear multiple times:{}'.format(np.sum(qids.value_counts()>1)))

plt.figure(figsize=(12,5))
plt.hist(qids.value_counts(),bins=50)
#conclusion 80000q(1) 5000q(2) 2000(3)
'''
plt.yscale('log',nonposy='clip')   #clip method will display it in sorted order
plt.title("log-histogram")
plt.xlabel('number of occurence of question')
plt.ylabel('number of question')
print()'''

#counting the no of character
all_ques_df["num_of_chars"]=all_ques_df["questions"].apply(lambda x:len(str(x)))
cnt_srs=all_ques_df['num_of_chars'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index,cnt_srs.values,alpha=0.8,color=color[0])
plt.ylabel('number of occurence',fontsize=12)
plt.xlabel('number of charactrer in question',fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
#CONCLUSION-NO INFO


#character
train_qs=pd.Series(train['question1'].tolist()+train['question2'].tolist()).astype(str)
test_qs=pd.Series(test['question1'].tolist()+test['question2'].tolist()).astype(str)

dist_train=train_qs.apply(len)
dist_test=test_qs.apply(len)
plt.figure(figsize=(15,10))
plt.hist(dist_train,bins=20,range=[0,200],color=pal[2],normed=True,label='train')
plt.hist(dist_test,bins=20,range=[0,200],color=pal[1],normed=True,alpha=0.5,label='test')
plt.title('normalized histogram of character')
plt.legend()
plt.xlabel('number of characters',fontsize=15)
plt.ylabel('probablity',fontsize=15)

#words
train_qs=pd.Series(train['question1'].tolist()+train['question2'].tolist()).astype(str)
test_qs=pd.Series(test['question1'].tolist()+test['question2'].tolist()).astype(str)

dist_train=train_qs.apply(lambda x:len(x.split(' ')))
dist_test=test_qs.apply(lambda x:len(x.split(' ')))
plt.figure(figsize=(15,10))
plt.hist(dist_train,bins=20,range=[0,200],color=pal[2],normed=True,label='train')
plt.hist(dist_test,bins=20,range=[0,200],color=pal[1],normed=True,alpha=0.5,label='test')
plt.title('normalized histogram of character')
plt.legend()
plt.xlabel('number of words',fontsize=15)
plt.ylabel('probablity',fontsize=15)

type(dfqa['question'].values)

#transform question by tf idf
mq1=TfidfVectorizer().fit_transform(dfqa['question'])
#5000*7003 sparce matrix for reducing dimension we use 3d t-sne
diff_encoding=mq1[::2]-mq1[1::2]

#3D t-sne enbedding
'''t-distributed stochastic neighbhor embedding is atechnique for dimensionality 
rediction particularly well suited for high dimensionality dataset.this technique 
can be implemented via barnes hut approximation,allowing it to be applied on large 
real world dataset'''

'''we will use t-sne to embed the tf-idf vector in three dimension and create interactive 
scatter plot'''


#STOP WORDS
import nltk
stop_words=nltk.corpus.stopwords.words()


pd.set_option("display.max_columns",10)
#remove chracter that are not letter or specal characte

def clean_sentence(val):
    regex=re.compile('([^\s\w]|_)+')
    sentence=regex.sub('',val).lower()
    sentence=sentence.split(" ")
    
    for word in list(sentence):
        if word in stop_words:
            sentence.remove(word)
            
    sentence=" ".join(sentence) #it will conver the list back to sentence
    return sentence

def clean_trainframe(df):
    "drop nan then apply 'clean_sentence function to question1 and question2"
    df=df.dropna(how='any')
    for col in ['question1','question2']:
        df[col]=df[col].apply(clean_sentence) #1st question 1 then question2 after cleaning comes under the column name col
         
    return df
 
df=clean_trainframe(train)
df.head(5)
#corpus will put  clean sentence into list    
def build_corpus(df):
    "create a list of list containing words from each sentence"
    corpus=[]
    for col in ['question1','question2']:
        for sentence in df[col]:
            word_list=sentence.split(" ")
            corpus.append(word_list)
            
    return corpus

corpus=build_corpus(df)
corpus[0:2]


from gensim.models import word2vec
model=word2vec.Word2Vec(corpus,size=100,window=20,min_count=200,workers=4)
model.wv['trump']
model.wv['kill']

len(model.wv.vocab)

#CREATE AND TSNE MODEL AND PLOT IT
def tsne_plot(model):
    labels=[]
    tokens=[]
    
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
        
    tsne_model=TSNE(perplexity=40,n_components=2,init='pca',n_iter=2500,random_state=23)
    new_values=tsne_model.fit_transform(tokens)
    
    x=[]
    y=[]

    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16,16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i],y[i]),
                     xytext=(5,2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
        
tsne_plot(model)

model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=500, workers=4)
tsne_plot(model)

from sklearn.decomposition import PCA
from matplotlib import py
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)

for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()


model.most_similar('trump')
model.most_similar('sex')
model.most_similar('girl')

from collections import Counter
import matplotlib.pyplot as plt
import operator
def eda(df):
    print("duplicate count=%s,non duplicate count =%s"
           %(df.is_duplicate.value_counts()[1],df.is_duplicate.value_counts()[0]))
    question_ids_combined=df.qid1.tolist() + df.qid2.tolist()
    print("unique questions =%s"%(len(np.unique(question_ids_combined))))
    question_ids_counter=Counter(question_ids_combined) #counter counts the number of occurence of each of the word
    sorted_question_ids_counter=sorted(question_ids_counter.items(),key=operator.itemgetter(1))
    question_appearing_more_than_once=[i for i in question_ids_counter.values() if i > 1]
    print("count of question appearing more than once=%s"%len(question_appearing_more_than_once))
eda(train)
 

import re
import gensim
from gensim import corpora
from nltk.corpus import stopwords

words=re.compile(r"\w+",re.I)
stopwords=stopwords.words('english')

def tokenize_questions(df):
    question_1_tokenized=[]
    question_2_tokenized=[]
    
    for q in df.question1.tolist():
        question_1_tokenized.append([i.lower() for i in words.findall(q) if i not in stopwords])
        
    for q in df.question2.tolist():
        question_2_tokenized.append([i.lower() for i in words.findall(q) if i not in stopwords])
        
    df["Question_1_tok"]=question_1_tokenized
    df["Question_2_tok"]=question_2_tokenized
    
    return df
def train_dictionary(df):
    questions_tokenized=df.Question_1_tok.tolist()+df.Question_2_tok.tolist()
    
    dictionary=corpora.Dictionary(questions_tokenized)
    dictionary.filter_extremes(no_below=5,no_above=0.5,keep_n=10000000)
    dictionary.compactify()
    
    return dictionary

df_train=tokenize_questions(train)
dictionary=train_dictionary(df_train)

print("no of words in the dictinary=%s" %len(dictionary.token2id))
    
def get_vectors(df,dictionary):
    #creating a corpus from dictionary
    question1_vec=[dictionary.doc2bow(text) for text in df.Question_1_tok.tolist()] 
    question2_vec=[dictionary.doc2bow(text) for text in df.Question_2_tok.tolist()]
    #creating a sparse matrix
    question1_csc=gensim.matutils.corpus2csc(question1_vec, num_terms=len(dictionary.token2id)) #num terms is the no of columns
    question2_csc=gensim.matutils.corpus2csc(question2_vec, num_terms=len(dictionary.token2id))

    return question1_csc.transpose(),question2_csc.transpose()
q1_csc,q2_csc=get_vectors(df_train,dictionary)

print(q1_csc.shape)
print(q2_csc.shape)

'''
cos(d1,d2)=(d1.d2)/||d1|| ||d2||
where indicates vector dot product
eg:-find the similarity between doc1 and doc2
d1=(5,0,3,0,2,0,0,2,0,0)
d2=(3,0,2,0,1,1,0,1,0,1)
d1.d2=5*3+0*0+3*2+0*0+2*1...=25
||d1||=5*5 + 0*0 + 3*2 ...=42
||d2||=3*3+0*0+2*2...=17
cos(d1,d2)=0.94

'''

from sklearn.metrics.pairwise import cosine_similarity as cs
def get_cosine_similarity(q1_csc,q2_csc):
    cosine_sim=[]
    for i,j in zip(q1_csc,q2_csc):
        sim=cs(i,j)
        cosine_sim.append(sim[0][0])
        
    return cosine_sim
cosine_sim=get_cosine_similarity(q1_csc,q2_csc)
print(len(cosine_sim))



from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.pipeline import Pipeline

np.random.seed(10)

def train_rfc(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=42)
    svm_models=[('svm',SVC(verbose=1,shrinking=False))]
    svm_pipeline=Pipeline(svm_models)
    svm_params={'svm__kernel':['rbf'],
                'svm__C':[0.01,0.1,1],
                'svm__gamma':[0.1,0.2,0.4],
                'svm__tot':[0.001,0.01,0.1],
                'svm__class_weight':[{1:0.8,0:0.2}] }
    
    rfc_models=[('rfc',RFC())]
    rfc_pipeline=Pipeline(rfc_models)
    rfc_params={'rfc__n_estimators' : [40],
                'rfc__max_depth' : [40]}
    
    lr_models=[('lr',LR(verbose=1))]
    lr_pipeline=Pipeline(lr_models)
    lr_params={'lr__C':[0.1,0.01],
               'lr__tol':[0.001,0.01],
               'lr__max_iter':[200,400],
               'lr__class_weight':[{1:0.8,0:0.2}]}
    
    
    gbc_models=[('gbc',GBC(verbose=1))]
    gbc_pipeline=Pipeline(gbc_models)
    gbc_params={'gbc__n_estimators':[100,200,400,800],
                'gbc__max_depth':[40,80,160,320],
                'gbc__learning_rate':[0.01,0.1]}

    grid=zip([svm_pipeline,rfc_pipeline,lr_pipeline,gbc_pipeline],
             [svm_params,rfc_params,lr_params,gbc_params])
    
    grid=zip([lr_pipeline],[lr_params])
    best_clf=None
    
    for model_pipeline,param in grid:
        temp=GridSearchCV(model_pipeline,param_grid=param,cv=4,scoring='f1')
        temp.fit(X_train,y_train)
        
        
        if best_clf is None:
            best_clf=temp
            
        else:
            if temp.best_score_ > best_clf.best_score_:
                best_clf = temp
                
    model_details={}
    model_details["CV Accuracy"]=best_clf.best_score_
    model_details["model parameters"]=best_clf.best_params_
    model_details["test data score"]=best_clf.score(X_test,y_test)
    model_details["f1 score"]=f1_score(y_test,best_clf.predict(X_test))
    model_details["confusion matrix"]=str(confusion_matrix(y_test,best_clf.predict(X_test)))
    
    return best_clf,model_details
    

X=np.array(cosine_sim).reshape(-1,1)
y=df_train.is_duplicate
    
clf,model_details=train_rfc(X,y)
    
    





    
    
    
    
    
    
    
    
    
    
                
                
            
    
    
















