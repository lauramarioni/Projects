#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importo le librerie necessarie
import numpy as np
import sklearn
import pandas as pd
import re
import nltk
import seaborn as sns

seed=123456


# In[2]:


import preprocessor as p


# In[3]:


#carico il dataset
df = pd.read_json('development.jsonl', lines=True)
X=df["full_text"]
y=df["class"]


# In[4]:


X


# In[5]:


tweet_df=df.copy()
tweet_df


# In[6]:


print('Dataset size:',tweet_df.shape)
print('Columns are:',tweet_df.columns)


# In[7]:


tweet_df.info()


# In[8]:


data= pd.DataFrame(X)
#df_fin  = pd.DataFrame(tweet_df[['full_text']])
data


# In[9]:


import string
string.punctuation


# In[10]:


data.full_text


# In[94]:


df["retweet_count"].mean()


# In[12]:


df["retweeted"].value_counts()


# In[13]:


df['favorite_count'][df['favorite_count'].notnull()].mean()


# In[28]:


a=df['coordinates'][df['coordinates'].notnull()]
val=[]
latitude_list=[]
longitude_list=[]
count2=0
for n,i in enumerate(a.index):
   # print(a[i]['coordinates'])
    val.append(a[i]['coordinates'])
    #count2=count2+1
    longitude_list.append(val[n][0])
    latitude_list.append(val[n][1])
#val,val[4][1]
longitude_list



# In[29]:


def Average(lst): 
    return sum(lst) / len(lst) 

lat_average = Average(latitude_list) 
lon_average = Average(longitude_list) 
lat_average, lon_average


# In[30]:


X.shape, y.shape


# In[31]:


numPos=0;
numNeg=0;
for i,el in enumerate(y):
    if (el==1):
        numPos=numPos+1
    else:
        numNeg=numNeg+1
        
numPos,numNeg


# In[32]:


#percentuale positive nel dataset
rapporto=[numPos*100/(numPos+numNeg)]
rapporto


# In[33]:


import matplotlib.pyplot as plt


# In[34]:


import plotly
import plotly.graph_objects as go
data=[go.Bar(x=y.value_counts().index,
            y=y.value_counts().values)]
layout = go.Layout(
    autosize=False,
    width=500,
    height=400,
    xaxis=dict(
        title='Gender'),
    yaxis=dict(
        title='#samples'
    ))

fig = go.Figure(data=data, layout=layout)
fig


# In[35]:


lengths=[]
count=0
for x in df.full_text:
    lengths.append(len(x))
    count=count+1


# In[36]:


#lengths = [len(x) for x in df.full_text]    
fig, ax = plt.subplots()
ax.set_xlim(0, 39930)
ax.set_ylim(0,350)

plt.title('Text lengths')
for i in range(count):
    plt.bar(i,lengths[i], color='b')
plt.savefig('lengths.png')


# In[37]:


nltk.download('stopwords')


# In[40]:


#Genero wordclouds per le recensioni positive
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords as sw

#stopwords = set(STOPWORDS)
#stopwords.update(["https","RT","I",",","@"])
stop_words = sw.words('english') 
new_stopwords = ["https","RT","I",",","@","We","Hashtag","How","many"]
stop_words.extend(new_stopwords)
stop_words = set(stop_words)

pos=df[df["class"]==1]
txt_pos=" ".join(review for review in pos["full_text"])
# Getting rid of the stopwords
clean_text = [word for word in txt_pos.split() if word not in stop_words]
# Converting the list to string
text = ' '.join([str(elem) for elem in clean_text])
text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)
text  = "".join([char for char in text if char not in string.punctuation])
text = re.sub('[0-9]+', '', text)

text


# In[39]:


# Generate a word cloud image
wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(text)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud for positive sentiment towards the movement.")
plt.axis("off")
plt.show()


# In[41]:


pos=df[df["class"]==0]
txt_pos=" ".join(review for review in pos["full_text"])
# Getting rid of the stopwords
clean_text = [word for word in txt_pos.split() if word not in stop_words]
# Converting the list to string
text = ' '.join([str(elem) for elem in clean_text])
text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)

text  = "".join([char for char in text if char not in string.punctuation])
text = re.sub('[0-9]+', '', text)


# In[42]:


wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(text)

# Display the generated image:
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud for negative sentiment towards the movement")
plt.axis("off")
plt.show()


# In[43]:


#Divido il mio dataset in train e test set
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.1, random_state=seed)


# In[44]:


X_train.shape


# In[45]:


#Calcolo numero recensioni positive e negative nel train
numPos=0;
numNeg=0;
for i,el in enumerate(y_train):
    if (el==1):
        numPos=numPos+1
    else:
        numNeg=numNeg+1
numPos,numNeg


# In[46]:


#Percentuale positive nel train set
rapporto=[numPos*100/(numPos+numNeg)]
rapporto


# In[47]:


#Calcolo numero recensioni positive e negative nel validation
numPos=0;
numNeg=0;
for i,el in enumerate(y_valid):
    if (el==1):
        numPos=numPos+1
    else:
        numNeg=numNeg+1
numPos,numNeg


# In[48]:


#Percentuale positive nel valid set
rapporto=[numPos*100/(numPos+numNeg)]
rapporto


# In[49]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as sw
import string
 

class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer=SnowballStemmer("italian")
        self.stopwords=stop_words
        self.punctuation=set(string.punctuation)
        
    def __call__(self, document):
        lemmas=[]
        re_digit=re.compile("[0-9]")#regular expression to filter digit tokens
        words = re.split(r'\W+', document)
        
        for t in words:
            lemma=t.strip()
            
            if lemma not in self.punctuation and len(lemma)>=2 and len(lemma)<16 and not re_digit.match(lemma)and lemma not in self.stopwords:
                    lemma=self.lemmatizer.stem(lemma)
                    lemmas.append(lemma)
                    
        return lemmas


# In[50]:


tokenizer = LemmaTokenizer()
vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 4), min_df=0.0005, encoding = 'utf-8')
X_tfidf = vectorizer.fit_transform(X_train)
X_tfidf


# In[51]:


# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[52]:


# load dataset
X = X_tfidf
Y = y_train
# prepare configuration for cross validation test harness
seed = 123456
# prepare models
models = []
models.append(('SGD', SGDClassifier(random_state=seed)))
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LSVC', svm.LinearSVC(random_state=seed)))
models.append(('DT', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(random_state=seed)))
#models.append(('NB', GaussianNB()))
#models.append(('RF', svm.LinearSVC(random_state=seed)))


# evaluate each model in turn
results = []
names = []
for name, model in models:
    cv_results = model_selection.cross_val_score(model, X.toarray(), Y, cv=5, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[53]:


# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[ ]:





# In[ ]:


#param_grid={"C":[0.25,0.1,0.5,0.6,0.7,0.75,0.8,1,2], "penalty":["l1","l2"], 'loss': ['hinge', 'squared_hinge'], 
#            'fit_intercept':['True','False']}
#clf=svm.LinearSVC(random_state = seed)
#gridsearch=GridSearchCV(clf,param_grid,scoring='f1_weighted',cv=5,n_jobs=-1, error_score=0.0)
#gridsearch.fit(X_tfidf,y_train)
#gridsearch.best_params_


# In[54]:


#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold


# In[56]:


param_grid = { 
    'n_estimators': [100, 300, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [75,100,165],
    'criterion' :['gini', 'entropy']
}


# In[57]:


from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


clf=RandomForestClassifier(random_state=seed)
gridsearch=GridSearchCV(clf,param_grid,scoring='accuracy',cv=5,n_jobs=-1, error_score=0.0)
gridsearch.fit(X_tfidf,y_train)
gridsearch.best_params_


# In[58]:


#utilizzo il classificatore random forest
clf = gridsearch.best_estimator_
clf.fit(X_tfidf, y_train)


# In[59]:


X_tfidf_valid = vectorizer.transform(X_valid)
X_tfidf_valid


# In[60]:


y_valid_pred=clf.predict(X_tfidf_valid)
y_valid_pred


# In[63]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_valid, y_valid_pred)


# In[64]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
conf_matrix=confusion_matrix(y_valid,y_valid_pred)

fig, ax = plt.subplots()
res = sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.xticks([0.5,1.5], [ 'Neg', 'Pos'],va='center')
plt.yticks([0.1,1.9], [ 'Neg', 'Pos'],va='center')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('confmatrix1.png',bbox_inches = 'tight')


# In[65]:


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

auc = roc_auc_score(y_valid,y_valid_pred)
auc


# In[66]:


#Carico il dataset da predire
df2 = pd.read_json('evaluation.jsonl', lines=True)


# In[67]:


X2=df2["full_text"]


# In[68]:


X_tfidf_test = vectorizer.transform(df2["full_text"])
X_tfidf_test


# In[69]:


#Effettuiamo la predizione sul notro dataset "evaluation"
y_test_pred=clf.predict(X_tfidf_test)
len(y)


# In[70]:


import csv
with open('results.csv', mode='w', newline='') as res_file:
    res = csv.writer(res_file, delimiter=',', quotechar='"')
    res.writerow(['Id', 'Predicted'])
    for i in range(y_test_pred.shape[0]):
        res.writerow([i, y_test_pred[i]])
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




