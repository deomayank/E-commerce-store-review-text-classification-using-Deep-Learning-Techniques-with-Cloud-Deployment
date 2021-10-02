import pandas as pd
scrapped_reviews = pd.read_csv('scrappedReviews.csv')   #Loading the scrapped reviws

Reviews = scrapped_reviews['Reviews']   #Taking only reviews for for further process
import numpy as np
Reviews = np.array(Reviews)

import pickle
file = open("pickle_model.pkl",'rb')    #rb means reading in binary mode.
recreated_model = pickle.load(file)
vectorizer = pickle.load(open('features.pkl','rb'))

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(vocabulary =vectorizer ) 

from sklearn.feature_extraction.text import TfidfTransformer
trasformer = TfidfTransformer()

predictions = recreated_model.predict(trasformer.fit_transform(vect.fit_transform(Reviews)))

pred = scrapped_reviews
pred['Positivity'] = predictions
pred.to_csv('Predictions.csv', index=False)

import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
cwd = os.getcwd()
if 'assets' not in os.listdir(cwd):
    os.makedirs(cwd+'/assets')
data = pd.read_csv('Predictions.csv')
chart_data = [data['Positivity'].value_counts()[1], data['Positivity'].value_counts()[0]]
plt.pie(chart_data, labels=['Positive reviews','Negetive reviews'], autopct='%.2f%%')
plt.savefig('assets/sentiment.png')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

word_list = []
for i in range(0, 7639):
    review = re.sub('[^a-zA-Z\s]', '', Reviews[i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]

    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    
    for j in range(0,len(review)):
        word_list.append(review[j])
        
word_list2 = pd.DataFrame()
word_list2['Words'] = word_list
top_100 = word_list2['Words'].value_counts()
k=top_100.head(100)
df = pd.DataFrame(k)
df.to_csv('Top 100 words.csv')
temp = pd.read_csv('Top 100 words.csv')
x = temp.iloc[:,0]

import wordcloud
from wordcloud import WordCloud, STOPWORDS
dataset = x.to_list()
str1 = ''
for i in dataset:
    str1 = str1+i
str1 = str1.lower()

stopwords = set(STOPWORDS)
cloud = WordCloud(width = 800, height = 400,
            background_color ='white',
            stopwords = stopwords,
            min_font_size = 10).generate(str1)
cloud.to_file("assets/wordCloud.png")