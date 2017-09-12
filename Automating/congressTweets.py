
# coding: utf-8

# In[1]:

# Download IMDB data from site -- http://ai.stanford.edu/~amaas/data/sentiment/
# p.234 of Python Machine Learning by Sebastian Raschka


# In[2]:

import os
import pandas as pd

pwd = os.getcwd()
print(pwd)
file = os.listdir(pwd)


# In[3]:

#################
# Start here    #
#################
#import pyprind

df = pd.read_csv(pwd+'/movie_data.csv', encoding='utf-8')
df.columns = ['review', 'sentiment']

print(df.shape)


# In[4]:

df.head(10)


# In[5]:

df.loc[3]


# # Prep Data

# In[6]:

# Bag of Word model
# 1. create a vocabulary of unique tokens (or words)
# 2. construct a feature vector for each document, features store count
#    of words per document

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer() #Instanstiate the count array

docs = np.array(['The sun is shining', 
                 'The weather is sweet',
                 'The sun is shining and the weather is sweet'])

bag = count.fit_transform(docs)

print(count.vocabulary_)
print(bag.toarray())


# In[7]:

# tf(t,d) - raw term frequencies (t: term, d: nos times term t appears in doc d)
# tf-idf(t,d) - term frequency inverse document frequency
# tf-idf = tf(t,d) * idf(t,x)  = tf(t,d) * log( [1+nd]/[1+df(d,t)] ) 


# In[8]:

# TfidTransformer
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer() #Instantiate Term Frequency invers

np.set_printoptions(precision=2)

print(tfidf.fit_transform(count.fit_transform(docs)).toarray()) #How much did the term appear in other documents?


# In[9]:

# so if the term "the" shows up lots of times, does that mean it's important?
# how can we make terms that shows up lots of times across documents, less important
# let's normalize by the times these terms show up across documents.

# employ : [nos of docs containing term "the" ]/[total nos of documents]

# if term appear often, give it less emphasis

# tf-idf(t,d) = tf(t,2)*(idf(t,d)+1)
# with idf(t,d) = log ([1+total nos of docs]/[1+nos of docs containing term t])


# In[10]:

#Reg functions...to get rid of HTML Tags and emoticons

import re
def preprocessor(text): 
# find '<' then anything not '>' [^>], [^>]* 0 or more prefix, then close with '>'    
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) 
    # eyes[:,;,=], optional nose [-], and mouth[),(,D,P)]
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    return text


# In[11]:

tmp = 'is ;) :) seven.<br /><br />Title (Brazil): Not Available'

print(preprocessor(tmp))
#print(preprocessor('</a>This :) is :( a test :-)!' ))
#print(re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', '</a>This :) is :( a test :-)!'))


# In[12]:

df['review'] = df['review'].apply(preprocessor) #use the apply method and send in the preprocessor function (applys the function to each row)


# In[13]:

df.shape


# In[14]:

df.tail(3)


# In[15]:

# p.242 Processing documents into tokens
# split the sentence/corpora into individual elements
def tokenizer(text):
    return text.split()


# In[16]:

tokenizer('running like running and thus they run')


# In[17]:

# word stemming, tranforming word into their root form
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[18]:

tokenizer_porter('running like running and thus they run')


# In[19]:

import nltk
nltk.download('stopwords')


# In[20]:

from nltk.corpus import stopwords
stop = stopwords.words('english')  # stop words have little meaning eg. a, is, and, has, etc. 
[w for w in tokenizer_porter('a runner likes running and runs a lot') 
 if w not in stop]


# # Let's Do Some ML

# In[21]:

# pg. 244 of Python Machine Learning by Sebastian Raschka
# Training a Logistic Regression model for document classification

#Test/Train
# (X,y)


X = df.loc[:, 'review'].values
y  = df.loc[:, 'sentiment'].values

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values

X_test  = df.loc[25000:, 'review'].values
y_test  = df.loc[25000:, 'sentiment'].values

print(y_test.shape)


# #Find the Best Parameters using automation!

# In[22]:

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# In[23]:

tfidf = TfidfVectorizer(strip_accents = None, 
                       lowercase = False)


# In[24]:

#Optimize to find best params

param_grid = [
              {'vect__ngram_range':[(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer], #, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [100]}, # 0.1, 1.0, 10.0, 100.0]},
              
              {'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer], #, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C':[100]} #[0.1, 1.0,10.0,100.0]}
                ]


# In[25]:

lr_tfidf = Pipeline([ ('vect', tfidf) ,
                      ('clf',  LogisticRegression(random_state=0))])



# In[26]:

gs_lr_tfidf = GridSearchCV( lr_tfidf, param_grid, #sends each subset to a different core
                          scoring = 'accuracy',
                          cv = 5, verbose = 1,
                          n_jobs = -1) # n_jobs -1 uses all computer cores


# In[27]:

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[28]:

#Takes like 60 seconds

gs_lr_tfidf.fit(X_train, y_train) 


# #Results

# In[29]:

print('The Best parameter set: %s' % gs_lr_tfidf.best_params_)


# In[30]:

print('CV Accuracy: %.3f'
     % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


# In[31]:

import pickle
filename = 'IMDB_model.pkl'
pickle.dump(gs_lr_tfidf, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)


# In[32]:

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[33]:

#Learning Curve, Takes Forever; Skip

#estimator = gs_lr_tfidf

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
#cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)


#title = "Learning Curve (LogReg, $\IMDB$)"
### SVC is more expensive so we do a lower number of CV iterations:
#plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)

#plt.show()



# # Random Forest

# In[34]:

from sklearn.ensemble import RandomForestClassifier

rf_clf=RandomForestClassifier(max_depth=None)
#tree_clf.fit(X,y)
print(rf_clf)


# In[35]:

rf_tfidf = Pipeline([ ('vect', tfidf) ,
                      ('clf',  rf_clf)])



# In[36]:

rf_tfidf.fit(X_train, y_train) 


# In[37]:

#from sklearn.model_selection import cross_val_score
#cv_scores = cross_val_score(rf_tfidf, X, y, cv=10)
#print(cv_scores)
#print(np.mean(cv_scores))


# # Importing Data - Tweet Time

# ##Import Model

# In[38]:

# load the model from disk
filename = 'IMDB_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
clf = loaded_model.best_estimator_
result = clf.score(X_test, y_test)
print(result)


# #Data Prep

# In[39]:

import datetime as dt
from datetime import date, timedelta

#Ceate a vaiarble: todays_tweets = 
today=dt.datetime.today().strftime("%Y-%m-%d")
yesterday = str(dt.date.today() - timedelta(1))
yesterday_json=(yesterday +  '.json')
url='https://alexlitel.github.io/congresstweets/data/'
url_json= (url + yesterday_json)
print(url_json)


# In[40]:

#Create Dates / A-axis
from datetime import date
d1 = date(2017, 6, 22)
d0 = date.today()
delta = d0 - d1
periods=delta.days
dates = pd.date_range('20170622', periods=periods)
print(dates)


# In[41]:

#Change Dates to string fo manipulation
datesStr=dates.strftime('%Y-%m-%d')


# In[42]:

#Cread new pivot table dfPlot
dfPlot = pd.DataFrame()
dfPlot['ScoreLR', 'scoresRF']=0
scoresLR=[]
scoresRF=[]


# #Loop to import and predict each day from URL

# In[43]:

import json
import urllib
import urllib.request


for date in datesStr:
    i=0
    date_json=date+'.json'
    url='https://alexlitel.github.io/congresstweets/data/'
    url_json=url+date_json
    data = urllib.request.urlopen(url_json).read()
    output = json.loads(data)
    dfJson = pd.DataFrame(output)
    predictLR = np.mean(clf.predict(dfJson['text']))
    predictRF = np.mean(rf_tfidf.predict(dfJson['text']))
    scoresLR=np.append(scoresLR,predictLR)
    scoresRF=np.append(scoresRF,predictRF)
    i+=1
    print('sucess '+date)



# In[44]:

dfPlot.head()
print(scoresLR, scoresRF)


# In[45]:

#set pictue
#Picture Cedits Alina Oleynik
from IPython.display import Image
Smiley=Image("Smiley.png")
Frowney=Image("Frowney.png")

if predictLR > .5:
    todaysFacePic = Smiley 
else:
    todaysFacePic = Frowney

# write image to png
fh = open('todaysFacePic.png','w')
fh.write('todaysFacePic.png')
fh.close()

#import urllib.request
#with urllib.request.urlopen('https://github.com/whs2k/whs2k.github.io/blob/master/Frowney.png?raw=true') as url:
#   facePic = url.read()
 
print('Congress Mood Today is:')
todaysFacePic


# In[46]:

#https://matplotlib.org/users/recipes.html

import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
thfont = {'fontname':'Tahoma'}

plt.clf()


fig, ax = plt.subplots(1)
ax.plot(dates, scoresLR)

# rotate and align the tick labels so they look better
fig.autofmt_xdate()

import matplotlib.dates as mdates
ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
plt.title('Congress Mood By Twitter')

plt.xlabel('Date',**thfont)
plt.ylabel('Sentiment',**thfont)
plt.savefig(pwd+'/todaysMood.svg', ppi=1200)
plt.show()
plt.close()



# # Add in Party Information - TBD

# In[47]:

#dfAccts = pd.read_csv(pwd+'/congressTwitters.csv')
#dfNames = pd.read_csv(pwd+'/congressNames.csv')
#dfNames.head()
#dfNames.drop[4]
#dfNames.dropna()#


# # Prepping and Automating this Script

# In[48]:

#Convert this notebook to a script
#$ ipython nbconvert --to script "congressTweets.ipynb"

#Then Execute
#$ python "congressTweets.py"




# In[49]:

#Automate itpi
#http://naelshiab.com/tutorial-how-to-automatically-run-your-scripts-on-your-computer/
#1. Create a new text file: 
#    #!/bin/sh
#    python python /Users/whs/Documents/DataJournalism/CongressionalTweets/whs2k.github.io/congressTwitter.py
#2. Save it with no extension
#3. Convert it to an excecutable 
#    chmod 755 command
#4. Set it up as an app in automater
#5. Make it an alert on calender


# In[50]:

# Automate a git file
#Last you need to git
#git add ..
#git commit -m "daily update"
#git push


# In[ ]:



