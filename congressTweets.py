
# coding: utf-8

# In[1]:

# Download data from site -- http://ai.stanford.edu/~amaas/data/sentiment/
# p.234
#import pyprind # may need to do >sudo easy_install pip, then >pip install pyprind --user
import pandas as pd
import os
#import pyprind


# In[2]:

pwd = os.getcwd()
print(pwd)
file = os.listdir(pwd)


# In[3]:

#################
# Start here    #
#################
#import pyprind
import pandas as pd
import os
pwd = os.getcwd()

df = pd.read_csv(pwd+'/movie_data.csv', encoding='utf-8')
df.columns = ['review', 'sentiment']

print(df.shape)


# In[4]:

df.head(5)


# In[5]:

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


# In[6]:

# tf(t,d) - raw term frequencies (t: term, d: nos times term t appears in doc d)
# tf-idf(t,d) - term frequency inverse document frequency
# tf-idf = tf(t,d) * idf(t,x)  = tf(t,d) * log( [1+nd]/[1+df(d,t)] ) 


# In[7]:

# TfidTransformer
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer() #Instantiate Term Frequency invers

np.set_printoptions(precision=2)

print(tfidf.fit_transform(count.fit_transform(docs)).toarray()) #How much did the term appear in other documents?


# In[8]:

# so if the term "the" shows up lots of times, does that mean it's important?
# how can we make terms that shows up lots of times across documents, less important
# let's normalize by the times these terms show up across documents.

# employ : [nos of docs containing term "the" ]/[total nos of documents]

# if term appear often, give it less emphasis

# tf-idf(t,d) = tf(t,2)*(idf(t,d)+1)
# with idf(t,d) = log ([1+total nos of docs]/[1+nos of docs containing term t])


# In[9]:

#Reg functions...to get rid of HTML Tags and emoticons

import re
def preprocessor(text): 
# find '<' then anything not '>' [^>], [^>]* 0 or more prefix, then close with '>'    
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) 
    # eyes[:,;,=], optional nose [-], and mouth[),(,D,P)]
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    return text


# In[10]:

tmp = 'is ;) :) seven.<br /><br />Title (Brazil): Not Available'

print(preprocessor(tmp))
#print(preprocessor('</a>This :) is :( a test :-)!' ))
#print(re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', '</a>This :) is :( a test :-)!'))


# In[11]:

df['review'] = df['review'].apply(preprocessor) #use the apply method and send in the preprocessor function (applys the function to each row)


# In[12]:

df.shape


# In[13]:

df.tail(3)


# In[14]:

# p.242 Processing documents into tokens
# split the sentence/corpora into individual elements
def tokenizer(text):
    return text.split()


# In[15]:

tokenizer('running like running and thus they run')


# In[16]:

# word stemming, tranforming word into their root form
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[17]:

tokenizer_porter('running like running and thus they run')


# In[18]:

import nltk
nltk.download('stopwords')


# In[19]:

from nltk.corpus import stopwords
stop = stopwords.words('english')  # stop words have little meaning eg. a, is, and, has, etc. 
[w for w in tokenizer_porter('a runner likes running and runs a lot') 
 if w not in stop]


# In[20]:

# pg. 244
# Training a Logistic Regression model for document classification
# (X,y)
#X_train = df.loc[:25000, 'review'].values
#y_train = df.loc[:25000, 'sentiment'].values

#X_test  = df.loc[25000:, 'review'].values
#y_test  = df.loc[25000:, 'sentiment'].values

X_train = df.loc[:2500, 'review'].values
y_train = df.loc[:2500, 'sentiment'].values

X_test  = df.loc[2500:5000, 'review'].values
y_test  = df.loc[2500:5000, 'sentiment'].values

print(y_test.shape)


# In[21]:

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# In[22]:

tfidf = TfidfVectorizer(strip_accents = None, 
                       lowercase = False)


# In[23]:

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


# In[24]:

lr_tfidf = Pipeline([ ('vect', tfidf) ,
                      ('clf',  LogisticRegression(random_state=0))])



# In[25]:

gs_lr_tfidf = GridSearchCV( lr_tfidf, param_grid, #sends each subset to a different core
                          scoring = 'accuracy',
                          cv = 5, verbose = 1,
                          n_jobs = -1) # n_jobs -1 uses all computer cores


# In[26]:

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[27]:

#Takes like 60 seconds

gs_lr_tfidf.fit(X_train, y_train) 


# In[28]:

print('The Best parameter set: %s' % gs_lr_tfidf.best_params_)


# In[29]:

print('CV Accuracy: %.3f'
     % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


# In[30]:

#df_20170628 = pd.read_json('2017-06-28.json')


# # Shiz getting real - Tweet Time

# In[31]:

import datetime as dt
from datetime import date, timedelta

#Ceate a vaiarble: todays_tweets = 
#today=dt.datetime.today().strftime("%m/%d/%Y")
today = dt.date.today().strftime("%Y-%m-%d")
yesterday = dt.date.today() - timedelta(1)
yesterday=yesterday.strftime("%Y-%m-%d")


# In[32]:

#Ceate a vaiarble: todays_tweets = 
today=dt.datetime.today().strftime("%Y-%m-%d")
yesterday_json=yesterday+'.json'
url='https://alexlitel.github.io/congresstweets/data/'
url_json=url+yesterday_json
#https://alexlitel.github.io/congresstweets/data/2017-07-04.json
print(url_json)


# In[33]:

#Create Dates / A-axis
from datetime import date
d1 = date(2017, 6, 22)
d0 = date.today()
delta = d0 - d1
periods=delta.days
dates = pd.date_range('20170622', periods=periods)

for date in dates:
    print(date)
print(dates)


# In[34]:

#Change Dates to string fo manipulation
datesStr=dates.strftime('%Y-%m-%d')


# In[35]:

#Cread new pivot dfPlot
dfPlot = pd.DataFrame()
dfPlot['Score']=0
#dfPlot['Date']=dates


# In[36]:

import json
import urllib
import urllib.request
data = urllib.request.urlopen(url_json).read()
output = json.loads(data)
dfJson = pd.DataFrame(output)
#print (output)
dfJson.head()

for date in datesStr:
    #dates=dates.strftime("%Y-%m-%d")
    date_json=date+'.json'
    url='https://alexlitel.github.io/congresstweets/data/'
    url_json=url+date_json
    data = urllib.request.urlopen(url_json).read()
    output = json.loads(data)
    dfJson = pd.DataFrame(output)
    predict = np.mean(clf.predict(dfJson['text']))
    #dfPlot=dfPlot.append(predict)
    dfPlot.loc[date]=predict
    #dfPlot[date] = predict#add to dfPlot
    #dfPlot.set_value(1, dfPlot[date], 'date')
    print('sucess'+date)



# In[37]:

dfPlot.head()


# In[38]:

#set pictue
#Picture Cedits Alina Oleynik
from IPython.display import Image
smiley=Image("Smiley.png")
frowney=Image("Frowney.png")

if predict > .5:
    facePic = smiley 
else:
    facePic = frowney
    
facePic

import urllib.request
with urllib.request.urlopen('https://github.com/whs2k/whs2k.github.io/blob/master/Frowney.png?raw=true') as url:
    facePic = url.read()
#I'm guessing this would output the html source code?
#print(s)
outfile = open('facePic.png','wb')
outfile.write(facePic)
outfile.close()
#facePic
 


# In[39]:

#X-Axis - Days
#max_year=df['fiscal_year'].max()
#min_year=df['fiscal_year'].min()
#years=np.linspace(min_year, max_year, (max_year-min_year+1))

import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
thfont = {'fontname':'Tahoma'}


#plt.plot(dates, dfPlot['Score'],'#daccc9', label='Congress Twitterr Sentiment')

#https://matplotlib.org/users/recipes.html

fig, ax = plt.subplots(1)
ax.plot(dates, dfPlot['Score'])
# rotate and align the tick labels so they look better
fig.autofmt_xdate()

# use a more precise date string for the x axis locations in the
# toolbar
import matplotlib.dates as mdates
ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
plt.title('Congress Mood By Twitter')

plt.xlabel('Date',**thfont)
plt.ylabel('Sentiment',**thfont)
plt.savefig(pwd+'/todaysMood.png')
plt.show()



# # Prepping and Automating this Script

# In[40]:

#Convert this notebook to a script
#$ ipython nbconvert --to script "congressTweets.ipynb"

#Then Execute
#$ python "congressTweets.py"




# In[41]:

#Automate it
#http://naelshiab.com/tutorial-how-to-automatically-run-your-scripts-on-your-computer/
#1. Create a new text file: 
#    #!/bin/sh
#    python python /Users/whs/Documents/DataJournalism/CongressionalTweets/whs2k.github.io/congressTwitter.py
#2. Save it with no extension
#3. Convert it to an excecutable 
#    chmod 755 command
#4. Set it up as an app in automater
#5. Make it an alert on calender


# In[42]:

# Automate a git file
#Last you need to git
#git add ..
#git commit -m "daily update"
#git push

