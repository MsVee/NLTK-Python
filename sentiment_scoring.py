
import os  # operating system commands
import re  # regular expressions
import nltk  # draw on the Python natural language toolkit
import pandas as pd  # DataFrame structure and operations
import numpy as np  # arrays and numerical processing
import matplotlib.pyplot as plt  # 2D plotting
import statsmodels.api as sm  # logistic regression
import statsmodels.formula.api as smf  # R-like model specification
import patsy  # translate model specification into design matrices
from sklearn import svm  # support vector machines
from sklearn.ensemble import RandomForestClassifier  # random forest
from langdetect import detect
from nltk.corpus import PlaintextCorpusReader
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import cross_validation
from nltk.collocations import *
import collections
from nltk.util import ngrams
import pdb
import pygal
from collections import Counter
from prettytable import PrettyTable
from BeautifulSoup import BeautifulSoup as bs


class MyObj(object):
    def __init__(self, num_loops):
        self.count = num_loops

    def go(self):
        for i in range(self.count):
            pdb.set_trace()
            print i
        return

def clean_tweet(tweet):

    more_stop_words = ['rt', 'cant','didnt','doesnt','dont','goes','isnt','hes','shes','thats','theres',\
					  'theyre','wont','youll','youre','youve', 'br', 've', 're', 'vs', 'goes','isnt',\
					  'hes', 'shes','thats','theres','theyre','wont','youll','youre','youve', 'br',\
                      've', 're', 'vs', 'this', 'i', 'get','cant','didnt','doesnt','dont','goes','isnt','hes',\
					  'shes','thats','theres','theyre','wont','youll','youre','youve', 'br', 've', 're', 'vs']
    
    # start with the initial list and add the additional words to it.
    stoplist = nltk.corpus.stopwords.words('english') + more_stop_words
    #stoplist = nltk.corpus.stopwords.words('english')

    # define list of codes to be dropped from document
    # carriage-returns, line-feeds, tabs
    codelist = ['\r', '\n', '\t']

    # insert a space at the beginning and end of the tweet
    tweet = ' ' + tweet + ' '
    tweet1=re.sub(r'[^\x00-\x7F]+',"", tweet)
    tweet2=re.sub(r'"', '', tweet1)
    tweet3 = re.sub(",", "", tweet2)
    tweet4 = re.sub("http[^\\s]+","", tweet3)
    
    # replace non-alphanumeric with space
    # temp1_tweet = re.sub('[^a-zA-Z]', '  ', tweet)
    # temp_tweet = re.sub('\d', '  ', temp1_tweet)

    for i in range(len(codelist)):
        stopstring = ' ' + codelist[i] + '  '
        temp_tweet = re.sub(stopstring, '  ', tweet4) #change to temp_tweet 
       
    # convert uppercase to lowercase
    temp_tweet = temp_tweet.lower()    

    # replace single-character words with space
    temp_tweet = re.sub('\s.\s', ' ', temp_tweet)

    # replace selected character strings/stop-words with space
    for i in range(len(stoplist)):
        stopstring = ' ' + str(stoplist[i]) + ' '
        temp_tweet = re.sub(stopstring, ' ', temp_tweet)

    # replace multiple blank characters with one blank character
    temp_tweet = re.sub('\s+', ' ', temp_tweet)    
    return(temp_tweet)

def word_freq_dist(tweet_words):
    word_freq = dict()

    for words in tweet_words:
        if (word_freq.has_key(words)):
            # This word already exists in the frequency dictionary, bump the count
            word_freq[words] += 1
        else:
            # insert the word into the frequency dictionary
            word_freq[words] = 1

    return word_freq

def plotMostFrequentWords(words, plot_file_name, plot_title):

    # compute a frequency distribution dictionary.
    word_freq_dict = word_freq_dist(words)

    # convert the dictionary into a sorted list.
    # lambda signifies an anonymous function. In this case, this function 
    # takes the single argument x and returns x[1] (i.e. the item at index 1 in x).
    # The values in the dictionary are in column [1]. lamda x: x[1] will sort the 
    # dictionary by the values of each entry within the dictionary; reverse=True
    # tells sorted to sort from largest to smallest instead of the default which is
    # smallest to largest.
    # see: http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
    
    freq_sorted_list = list()
    freq_sorted_list = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
    #freq_sorted_list[0][0] gives most frequent word
    #freq_sorted_list[0][1] gives count for that word

    # print the top 15 words and their counts
    print('Top 15 words in terms of frequency: ')

    max_num = 15
    if (len(freq_sorted_list) < max_num):
        max_num = len(freq_sorted_list)

    for i in range(max_num):
        print('index: ', i, ' words: ', freq_sorted_list[i][0],
              ' count: ', freq_sorted_list[i][1])

    #print('\n')
    
    # convert the sorted list into a data frame so that we can plot
    freq_sorted_df =  pd.DataFrame(freq_sorted_list, columns=['Word', 'Count'])

    #print freq_sorted_df.head()

    freq_sorted_word_chart = freq_sorted_df[:15].plot(kind='bar', x='Word', y='Count',
                                                      title = plot_title)
    
    freq_sorted_word_chart.set_ylabel('Word Count')
    freq_sorted_word_chart.set_xlabel('')
    freq_sorted_word_chart.legend().set_visible(False)
    plt.savefig((plot_file_name), bbox_inches = 'tight', edgecolor='b', orientation='landscape', papertype=None, format=None, 
    transparent=True)  # plot to file

    # clear the figure
    plt.clf()
    
    return freq_sorted_list

#Define directory and file with all tweets to be used
dir=('C:\\Users\\ecoker\\Documents\\Educational\\capstone\\')
twitter_df=pd.read_csv(dir + 'chicagobears120614.csv')
#clean up all tweets
cleaned_tweets = list()
review_tweets = twitter_df.status_text  

for line in review_tweets:
        cleaned_tweet = clean_tweet(line)    
        cleaned_tweets.append(cleaned_tweet)

aggregate_corpus = nltk.Text(cleaned_tweets)


#look at freq of 2-word collocations
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder=nltk.collocations.BigramCollocationFinder.from_words(cleaned_tweets)
bigramslist=[]
bigramslisttop=[]
bigramslist = finder.ngram_fd.keys()
bigramslisttop = finder.nbest(bigram_measures.pmi,10)
#bigramslisttop = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:15]
aggregate_corpus_bi = nltk.Text(bigramslist)
# print 'bigrams top created' 
# for word in bigramslisttop:
#     pt = PrettyTable(field_names =[word, 'Count'])
#     c = Counter(word)
#     [ pt.add_row(kv) for kv in c.most_common()[:10] ]
#     pt.align[word], pt.align['Count'] = 'l', 'r'
#     # print pt

#     #look at freq of 3-word collocations
# trigram_measures = nltk.collocations.TrigramAssocMeasures()
# finder=nltk.collocations.TrigramCollocationFinder.from_words(cleaned_tweets)
# trigramslist=[]
# trigramslisttop=[]
# trigramslist = finder.ngram_fd.keys()
# trigramslisttop = finder.nbest(trigram_measures.pmi,10)
# # trigramslisttop = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:15]
# aggregate_corpus_tri = nltk.Text(trigramslist)
# print 'trigrams top created' 


###################
# Use Python collection for counting frequency OF USERS
user_count = Counter()
retweet_count = Counter()
for index, row in twitter_df.iterrows():
    user_count[row['screen_name'] ] += 1
    retweet_count[row['retweet_count'] ] += 1

# Prepare the svg Plot

barplot = pygal.HorizontalBar( style=pygal.style.SolidColorStyle )

topnum = 10
for i in range(topnum):
    barplot.add( user_count.most_common(topnum)[i][0], \
              [ { 'value': user_count.most_common(topnum)[i][1], \
                  'label': user_count.most_common(topnum)[i][0]} ] )
barplot.config.title = barplot.config.title= "Top " + str(topnum) + " Most Prolific Tweeters"
barplot.config.legend_at_bottom=True
barplot.render_to_file("Top_Tweeters.svg")

################ Tweets with RT count > 10
count = Counter([i for i in cleaned_tweets])

frdf = []
for i,j in count.iteritems():
    if j > 10:
        frdf.append([j, i])

df1 = pd.DataFrame(frdf, index=None, columns=["Count", "Tweet"])
df1.sort(columns="Count", inplace=True, ascending=False)

# for i,j,k in df1.itertuples():
#     print j,"\t", k



# Get the source field from each tweet
# Reduce the source and count
src=Counter(twitter_df.source)

# Convert the "Counter" container to Pandas dataframe for easy manipulation
frame = []
for i,j in src.iteritems():
    match=re.match(r"^.*\">(.*)\<.*$", i)
    frame.append( [j, match.group(1)])
sourcedf = pd.DataFrame(frame, columns=["COUNT", "SOURCE"])

# A lookup table to normalize the data in the containers we want
#   - all iOS Platforms (iPad, iPhone et. al. goes into iOS etc.)
sourcelookup = { "web": "Web",                              "Twitter for iPhone": "iOS",
                "Twitter for Android": "Android",           "TweetDeck": "TweetDeck",
                "Tweetbot for iOS": "iOS",                  "Twitter for iPad": "iOS",
                "Twitter for Mac": "Mac",                   "Tweetbot for Mac": "Mac",
                "Twitter for Android Tablets": "Android",   "Twitterrific": "iOS",
                "iOS": "iOS",                               u"Plume\xa0for\xa0Android": "Android",
                "YoruFukurou": "Mac",                       "TweetCaster for Android": "Android",
                "Guidebook on iOS": "iOS",                  "Twitter for Android": "Android",
                "UberSocial for iPhone": "iOS",             "Twitterrific for Mac": "Mac"
                }


# A helper function for looking up the table defined above
def translate(txt):
    try:
        return sourcelookup[txt]
    except KeyError:
        return "Other"

# Create a new column with normalized field
sourcedf['NSOURCE']=sourcedf.SOURCE.apply(lambda x: translate(x))

# Groupby the normalized field "NSOURCE"
grouped = sourcedf.groupby(by=["NSOURCE"])

# Create the chart (PieChart)
chart = pygal.Pie( style=pygal.style.SolidColorStyle )

for i in grouped.groups.iteritems():
    chart.add( i[0], grouped.get_group(i[0]).COUNT.tolist() )

chart.config.title="Twitter Source for PyData-SV Users"
chart.render_to_file('pie_chart_twitter_usersource.svg')


### Time for Sentiment scoring
# read in positive and negative word lists from Hu and Liu (2004). Basis for scoring without training
with open('Hu_Liu_positive_word_list.csv','rt') as f:
    positive_word = f.read().split() 
with open('Hu_Liu_negative_word_list.csv','rt') as f:
    negative_word = f.read().split()  

# switch to new Twitter-focused lexicons
with open('C:\\Users\\ecoker\\Documents\\Projects\\SocialMining\\unigrams-pos.csv', 'rt') as f:
    positive_lex=f.read().split()
with open('C:\\Users\\ecoker\\Documents\\Projects\\SocialMining\\unigrams-neg.csv', 'rt') as f:
    negative_lex=f.read().split()

positive_word_list=positive_word + positive_lex
negative_word_list=negative_word + negative_lex

positive_list = []
for line in positive_word_list:
        cleaned_word_p = clean_tweet(line)    
        positive_list.append(cleaned_word_p)
positive_lists=set(positive_list)
negative_list = []
for line in negative_word_list:
        cleaned_word_n = clean_tweet(line)    
        negative_list.append(cleaned_word_n)
negative_lists=set(negative_list)
   

def score_positive(text):    
    positive = [w for w in text.split() if w in positive_lists]
    total = [w for w in text.split()]
    return 100 * len(positive)/len(total)

def score_negative(text):    
    negative = [w for w in text.split() if w in negative_lists]
    total = [w for w in text.split()]
    return 100 * len(negative)/len(total)



#take a look at ngrams, the next type to be used
# convert the sorted list into a data frame so that we can plot
bi_df =  pd.DataFrame(bigramslisttop, columns=['Word', 'Count'])
bigram_chart = bi_df.plot(kind='bar', x='Word', y='Count', title = 'bigrams')
bigram_chart.set_ylabel('Word Count')
bigram_chart.set_xlabel('')
bigram_chart.legend().set_visible(False)
plt.savefig(('bigrams'), bbox_inches = 'tight', edgecolor='b', orientation='landscape', papertype=None, format=None, transparent=True)  # plot to file

trip_df =  pd.DataFrame(trigramslisttop, columns=['Word', 'Count'])
trigram_chart = trip_df.plot(kind='bar', x='Word', y='Count',title = 'trigrams')
trigram_chart.set_ylabel('Word Count')
trigram_chart.set_xlabel('')
trigram_chart.legend().set_visible(False)
plt.savefig(('trigrams'), bbox_inches = 'tight', edgecolor='b', orientation='landscape', papertype=None, format=None, transparent=True)  # plot to file

def compute_scores(tweets):
    # use the complete word lists for POSITIVE and NEGATIVE measures
    # to score all documents in a corpus or list of documents
    positive = []
    negative = []
    for word in tweets:
        positive.append(score_positive(word)) 
        negative.append(score_negative(word)) 
    return(positive, negative)


positive, negative = compute_scores(aggregate_corpus) # or trigramslist or bigramslist or cleaned_tweets

data = {'positive':positive, 'negative':negative, 'cleaned_tweets':cleaned_tweets}
dataframe=pd.DataFrame(data)

dataframe['sentiment']=(dataframe['positive']/100)-(dataframe['negative']/100)

mean = dataframe.sentiment.mean
print 'mean sentiment is ', mean

columns=['positive', 'negative']
dataframe=dataframe.drop(columns, axis=1)

new_twitter_df=pd.concat([dataframe, twitter_df], axis=1)

# Plot the freq dist for the full corpus    
# plot a bar chart for top words in terms of counts
print('Get the top 15 words: ')
full_plot_file_name = dir + 'full_review_word_count.png'
plot_title = 'full_review_word_count'
full_sort = plotMostFrequentWords(aggregate_corpus, full_plot_file_name, plot_title)

# Plot the freq dist for the positive tweet words    
# plot a bar chart for top words in terms of counts
positive_corpus = nltk.Text(positive)
print('Get the top 15 positive words: ')
pos_plot_file_name = dir + 'pos_review_word_count.png'
plot_title =  'pos_review_word_count'
positive_sort = plotMostFrequentWords(positive_corpus, pos_plot_file_name, plot_title)

# Plot the freq dist for the negative tweet words   
# plot a bar chart for top words in terms of counts
negative_corpus = nltk.Text(negative)
print('Get the top 15 negative words: ')
neg_plot_file_name = dir + 'neg_review_word_count.png'
plot_title = '_neg_review_word_count'
negative_sort = plotMostFrequentWords(negative_corpus, neg_plot_file_name, plot_title)


#Finally do the modeling using classification models and predict sentiment
# sample=pd.read_csv(dir + 'sample_tweets_coded.csv')
# end_df=pd.merge(new_twitter_df, sample, how='left', right_index=True,left_index=True, on=None)

# # vectorize tweets for machine learning and remove stopwords
# vectorizer = CountVectorizer(min_df=1, stop_words='english')
# vector_data = vectorizer.fit_transform(end_df['cleaned_tweets'])
# # select only hand scored tweets for model training/evaluation
# scored_data = vector_data[end_df[end_df['Score'].isnull() == False].index]
# # create testing/training sets
# x_train, x_test, y_train, y_test = cross_validation.train_test_split(scored_data,
# end_df[end_df['Score'].isnull() == False]['Score'],
# test_size = 0.2, random_state = 0)
# print end_df.summary()

# # logistic regression classifier
# lr_clf = LogisticRegression()
# lr_clf = lr_clf.fit(x_train, y_train)
# lr_predicted = lr_clf.predict(x_test)
# # print classification report
# target_names = ['not postive','positive']
# print 'Logistic Regression Classification Report:'
# print (classification_report(y_test, lr_predicted, target_names = target_names))

# # support vector machine classifier
# from sklearn.linear_model import SGDClassifier
# svm = SGDClassifier()
# svm = svm.fit(x_train, y_train)
# svm_predicted = svm.predict(x_test)
# print 'Support Vector Machine Classification Report:'
# print (classification_report(y_test, svm_predicted, target_names = target_names))

# # naive bayes classifier
# from sklearn.naive_bayes import MultinomialNB
# nb_clf = MultinomialNB()
# nb_clf = nb_clf.fit(x_train, y_train)
# nb_predicted = nb_clf.predict(x_test)
# print 'Naive Bayes Classification Report:'
# print (classification_report(y_test, nb_predicted, target_names = target_names))

# # decided to use the output from the logistic regression
# # append results to data frame and save
# end_df['predicted_sentiment'] = lr_clf.predict(vector_data)



# end_df['positive_probability'] = lr_clf.predict_proba(vector_data)[:,1]

# end_df['negative_probability'] = lr_clf.predict_proba(vector_data)[:,0]

# end_df.to_csv(dir+'final_data120214.csv')

