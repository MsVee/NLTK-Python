from __future__ import division
import os  # operating system commands
import pandas as pd
import numpy as np
from numpy import log #F test
from sklearn import metrics  
from sklearn.tree import DecisionTreeClassifier  # CART Classifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt  # 2D plotting
import statsmodels.api as sm  # logistic regression
import statsmodels.formula.api as smf  # R-like model specification
from patsy import dmatrices  # translate model specification into design matrices
from sklearn import svm  # support vector machines
from sklearn.ensemble import RandomForestClassifier  # random forest
import pdb
from sklearn import pipeline
from sklearn import cross_validation
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.preprocessing import Imputer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA

# class for debugging errors
class MyObj(object):
    def __init__(self, num_loops):
        self.count = num_loops

    def go(self):
        for i in range(self.count):
            pdb.set_trace()
            print i
        return

dir=('C:\\Users\\ecoker\\Documents\\Projects\\Twitter\\Python-NLTK-and-Twitter\\')

twitter_df = pd.read_csv(dir + 'twitter_df.csv')
twitter_df.source = twitter_df.source.fillna('Unknown')
twitter_df.time_zone = twitter_df.time_zone.fillna('Unknown')
twitter_df.location = twitter_df.location.fillna('Not_provided')
twitter_df.to_csv('dataframe.csv', index=False, parse_dates=['tmstamp'])
# Use the aggregate DataFrame to perform Linear Regression on Response of Retweet Counts
train, test = train_test_split(twitter_df, test_size=.3, random_state=0)

testing = pd.DataFrame(test, columns = ['index', 'follower_count', 'friend_count', 'geo', 'location', 'name', 'place', 'retweet_count', 'screen_name', 'source', 'status_id', 'status_text', 'time_zone', 'timestamp', 'surge_pricing', 'free_rides', 'promo', 'driver', 'food', 'controversy', 'regulations'])
training = pd.DataFrame(train, columns = ['index', 'follower_count', 'friend_count', 'geo', 'location', 'name', 'place', 'retweet_count', 'screen_name', 'source', 'status_id', 'status_text', 'time_zone', 'timestamp', 'surge_pricing', 'free_rides', 'promo', 'driver', 'food', 'controversy', 'regulations'])
# training = training.drop(['unnamed0', 'geo', 'name' 'place', 'screen_name', 'status_id'], axis=1)
# testing = testing.drop(['unnamed0', 'geo', 'name' 'place', 'screen_name', 'status_id'], axis=1)
# training.to_csv(dir + 'training.csv')
# testing.to_csv(dir + 'testing.csv')
# y, X = dmatrices('retweet_count ~ surge_pricing + free_rides + promo + driver + food + controversy + regulations', data=twitter_df, return_type='dataframe')
# # Define the model from above Patsy-created variables, using Statsmodels
# print sm.OLS(y,X).fit().summary()
# print sm.OLS(y,X).fit().params
# print 'r sqd is : ', sm.OLS(y,X).fit().rsquared
# rainbow=sm.stats.linear_rainbow(sm.OLS(y,X).fit())
# print 'Rainbow Test for Linearity is ', rainbow
# y_hat, X_hat = dmatrices('retweet_count ~ surge_pricing + free_rides + promo + driver + food + controversy + regulations', data=testing, return_type='dataframe')
# y_pred = sm.OLS(y,X).fit().predict(X)
# twitter_df['retweet_pred'] = pd.Series(y_pred)

# Apply decision tree and/or CART classification on Retweet Count
model = DecisionTreeClassifier()
training['follower_count_l'] = training['follower_count'].replace(0, 1e-6)
training['follower_count_l'] = np.log(training['follower_count_l'])
testing['follower_count_l'] = testing['follower_count'].replace(0, 1e-6)
testing['follower_count_l'] = np.log(testing['follower_count_l'])
training['friend_count_l'] = training['friend_count'].replace(0, 1e-6)
training['friend_count_l'] = np.log(training['friend_count_l'])
testing['friend_count_l'] = testing['friend_count'].replace(0, 1e-6)
testing['friend_count_l'] = np.log(testing['friend_count_l'])
numeric_cols = ['friend_count', 'follower_count']
# X_num_tr = training[list(numeric_cols)].as_matrix().astype(np.float)
# X_num_te = testing[list(numeric_cols)].as_matrix().astype(np.float)
cat_cols = ['time_zone', 'timestamp', 'location', 'source', 'surge_pricing', 'free_rides', 'promo', 'driver', 'food', 'controversy', 'regulations']
columns = cat_cols + numeric_cols
# num_train = np.hstack((tr_foll, tr_friend))
# num_test = np.hstack((te_foll, te_friend))
# cattr = training[list(cat_cols)]
# # cat_tr = cattr.T.to_dict().values()
# catte = testing[list(cat_cols)]
# cat_te = catte.T.to_dict().values()
# vectorizer=DV(sparse=False)
# categ_train = vectorizer.fit_transform(cat_tr)
# categ_test = vectorizer.fit_transform(cat_te)
feat_num = training[list(numeric_cols)]
scaler = StandardScaler()
numx = scaler.fit_transform(feat_num)
features_cat = training[list(cat_cols)]
X = np.hstack((numx + features_cat))
t_features = {}
t_features = X.to_dict
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
vector = vec.fit_transform(t_features).toarray()
# features_x = sparse.csr_matrix(features.values)
# imp = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
# imp.fit(categ_train)
# categ_train = imp.transform(categ_train)
# print categ_train.shape
# print X_num_tr.shape
# print categ_test.shape
# print X_num_te.shape
# features_train = np.hstack((categ_train, X_num_tr))
# features_test = np.hstack((categ_test, X_num_te))


# vectorizer=DV(sparse=True)
# features_train = vectorizer.fit_transform(features_train)
# features_test = vectorizer.fit_transform(features_test)
# features_train = np.hstack((features_train, numtr))
# features_test = np.hstack((features_test, numte))
training['retweet_count'] = training['retweet_count'].replace(0, 1e-6)
testing['retweet_count'] = testing['retweet_count'].replace(0, 1e-6)
target_train = np.log((training['retweet_count']).values)
target_test = np.log((testing['retweet_count']).values)


model.fit(vector, target_train)
print(model)
# make the predictions
expected = target_train
predicted = model.predict(vector)
#check the fit of predictions
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
