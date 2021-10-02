import pandas
df = pandas.read_csv("balanced_reviews.csv")

df.shape
df.columns.tolist()
df.head()
df['reviewText'].head()
df['reviewText'][0]
df['overall'].unique()
df['overall'].value_counts()

# Now we drop all those rows which have at least one missing value.
df.isnull().any(axis = 0)
df.dropna(inplace = True)

# Now we drop all the reviews with rating 3 since it won't help in analysis.
df['overall'].value_counts()
df['overall'] != 3
df  = df[df['overall'] != 3]
df['overall'].value_counts()

# Creating a new column Positivity where
# 1 means Rating>3
# 0 means Rating<3
import numpy as np
df['Positivity']  = np.where(df['overall'] > 3, 1, 0)
df['Positivity'].value_counts()

# Splitting data for training and testing
features = df['reviewText']
labels = df['Positivity']
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)

# Converting features_train data to numeric representation (Feature Extraction)
'''
# Method 1: Count vectorizer
# CountVectorizer is used to encode ie convert to numeric codes the text passed to it word by word.
#   Eg - The quick brown fox jumped over the lazy dog. - {'dog': 1, 'fox': 2, 'over': 5, 'brown': 0, 'quick': 6, 'the': 7, 'lazy': 4, 'jumped': 3}
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer().fit(features_train)    # To fit data means to find its mean and standard deviation.
len(vect.get_feature_names())   # We found how any unique values are present in vect.
features_train_vectorized = vect.transform(features_train)      # After fitting we go for transformation which means converting numeric data to vales between 0 to 1.
vect.get_feature_names()[16000:16010]       # We can know all these words and can access them through the []
'''

# Method 2: TFIDF (more precise and relevent here)
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df = 5).fit(features_train)  # min_df = 5 means ignore terms/words that appear in less than 5 documents
len(vect.get_feature_names())
features_train_vectorized = vect.transform(features_train)

# Visuaization - Creating a frequency distribution graph which automatically shows top 50 words and their frequency in descending order.
from yellowbrick.text import FreqDistVisualizer
vocab = vect.get_feature_names()
visualizer = FreqDistVisualizer(features = vocab, orient = 'v')
visualizer.fit(features_train_vectorized)
visualizer.show()

# Visualization - Creating the same type of frequency distribution graph but this time with only two columns this time which are positive and negetive and thier frequency on y axis.
from yellowbrick.target import ClassBalance
visualizer = ClassBalance(labels = ['Negative','Positive'])
visualizer.fit(labels)
visualizer.show()

# Building the classifier/model
# There are various types of classifiers namely - binary classifier , SVC, kNN, Naive Bayes, Logistic Regression , DT, RF.
# Here we are going to use Logistic Regression.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(features_train_vectorized,labels_train)
# 0 -> negative
# 1 -> positive
predictions = model.predict(vect.transform(features_test))

# Creating confusion matrix to check details of accuracy.
from sklearn.metrics import confusion_matrix
confusion_matrix(labels_test, predictions)

# Finding the accuracy of our model after .
from sklearn.metrics import accuracy_score
accuracy_score(labels_test, predictions)

# Predicting in form of probability
model.predict_proba(vect.transform(features_test))
# predict - predicts as either 0 or 1.
# predict_proba - predicts exact probablity in decimal from 0 to 1.

# Visualizing the confusion matrix (unnecessary)
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(model, classes = [0,1])
cm.score(vect.transform(features_test),  labels_test)
cm.show()

#Team A - Creates the model
import pickle
file  = open("pickle_model.pkl","wb")   #wb means writing in binary mode.
pickle.dump(model, file)

#Team B - Uses the already created model for prediction
file = open("pickle_model.pkl",'rb')    #rb means reading in binary mode.
recreated_model = pickle.load(file)
preds = recreated_model.predict(vect.transform(features_test))
from sklearn.metrics import accuracy_score
accuracy_score(labels_test, preds)

# Now we also save the vectorizer because we transformed the data before building and testing the model. Without transforming the input data, the model will show an error that it is getting unexpected inputs.
vocab_file = open('features.pkl','wb')
pickle.dump(vect.vocabulary_, vocab_file)