import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import dataframe_image as dfi

# import data
corpus = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])

# create train and test set
x_train, x_test, y_train, y_test = train_test_split(corpus.message, corpus.label, test_size=0.2, random_state=53)

# set up vectorizers
## ... using bag of words
## note: default of param token pattern of CountVectorizer (= r”(?u)\b\w\w+\b”) removes punctuations 
bow_vectorizer = CountVectorizer(stop_words='english', lowercase=True) # init CountVectorizer object with set up to erase stopwords

## ... using tfidf 
tfidf_vectorizer = TfidfVectorizer(stop_words="english", lowercase=True) 

## ... using bag of N-Gram (n=2) 
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), lowercase=True)

## ... using bag of N-Gram (n=1 and 2) 
one2twoGram_vectorizer = CountVectorizer(ngram_range=(1, 2), lowercase=True)

## summarize vectorizers
vectorizers = []
vectorizers.append(("bow", bow_vectorizer))
vectorizers.append(("tfidf", tfidf_vectorizer))
vectorizers.append(("bigram", bigram_vectorizer))
vectorizers.append(("1-2-gram", one2twoGram_vectorizer))

# create, train and test naive baysian spam classifier for every vectorization approach
nb_results = [] # variable storing peroformance indicators to compare naive baysian models' performance depending on vectorization method
for name, vectorizer in vectorizers:

    # vectorize train and test data
    x_train_vectorized = vectorizer.fit_transform(x_train.values)
    x_test_vectorized = vectorizer.transform(x_test.values)

    # set up naive baysian model
    nb = MultinomialNB()

    # train naive baysian model
    nb.fit(x_train_vectorized, y_train)

    # test model
    pred = nb.predict(x_test_vectorized)

    # evaluate test
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, pos_label="spam")
    cm = confusion_matrix(y_test.values, pred, labels=['spam', 'ham'])

    # store results
    nb_results.append((name, accuracy, f1, cm[0][0], cm[0][1], cm[1][0], cm[1][1]))

# comparison of naive baysian spam classifier's performance depending on vectorization method
nb_performance_comparison = pd.DataFrame(nb_results)
nb_performance_comparison.columns = ["vectorizer", "accuracy", "f1", "true-positive", "false-positive", "false-negative", "true-negative"]
dfi.export(nb_performance_comparison, "documentation/tables_as_image/nb_performance_comparison.png")

# spam classifier based on a feed forward neural network
feed_forward_results = [] # variable storing peroformance indicators to compare feed forward neural net's performance depending on vectorization method
for name, vectorizer in vectorizers:
    # vectorize train and test data
    x_train_vectorized = vectorizer.fit_transform(x_train.values)
    x_test_vectorized = vectorizer.transform(x_test.values)

    # set up feedforward neural net classifier
    mlp = MLPClassifier(random_state=1, max_iter=300)
    
    # train neural net
    mlp.fit(x_train_vectorized, y_train)

    # test neural net
    pred = mlp.predict(x_test_vectorized)

    # evaluate test
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, pos_label="spam")
    cm = confusion_matrix(y_test.values, pred, labels=['spam', 'ham'])

    # store results
    feed_forward_results.append((name, accuracy, f1, cm[0][0], cm[0][1], cm[1][0], cm[1][1]))

# comparison of naive baysian spam classifier's performance depending on vectorization method
feed_forward_performance_comparison = pd.DataFrame(feed_forward_results)
feed_forward_performance_comparison.columns = ["vectorizer", "accuracy", "f1", "true-positive", "false-positive", "false-negative", "true-negative"]
dfi.export(feed_forward_performance_comparison, "documentation/tables_as_image/feed_forward_neural_net_performance_comparison.png")
