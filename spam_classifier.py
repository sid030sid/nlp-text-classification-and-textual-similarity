# to do: 
#continue wth task 3: list set up of hyper param of neural net (ideally like in lecture mentioned); compute f1 score; train and test nd and mlp models several times to have solid results for comparison
#task 4: in a clean way and in its on py file
#task 1 + write report


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# import data
corpus = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])

# task 2: pre-processing: 
# tokenization --> done by default through param token_pattern during init of vectorizer object
# remove stop words, --> done by setting param stop_words to english during init of vectorizer object. Alternatively, param max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms.
# punctuations --> done by default set up of param token_pattern during init of vectorizer object
# and white spaces as well as --> done by default set up of param token_pattern during init of vectorizer object
# normalise tokens by making all tokens lower case --> done by setting param lowercase to true during init of vectorizer object
# (source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

# create train and test set
x_train, x_test, y_train, y_test = train_test_split(corpus.message, corpus.label, test_size=0.2, random_state=53)

# set up vectorizers
## ... using bag of words --> requires following pre-processing according to lecture: Lowercasing, Punctuation removal, Word tokenization, Stop word removal
## note: default token pattern (= r”(?u)\b\w\w+\b”) removes punctuations 
bow_vectorizer = CountVectorizer(stop_words='english', lowercase=True) # init CountVectorizer object with set up to erase stopwords

## ... using tfidf --> requires following pre-processing according to lecture: Lowercasing, Punctuation removal, Word tokenization, Stop word removal
tfidf_vectorizer = TfidfVectorizer(stop_words="english", lowercase=True) # add eventually param: max_df=0.7 --> this erases additional stop words or words that occure to often accross all documents

## ... using bag of N-Gram (n=2) --> requires following pre-processing according to lecture:  Lowercasing, Punctuation removal, Word tokenization
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), lowercase=True)

## ... using bag of N-Gram (n=1-2) 
one2twoGram_vectorizer = CountVectorizer(ngram_range=(1, 2), lowercase=True)

## summarize vectorizers
vectorizers = [
    ["bow", bow_vectorizer], 
    ["tfidf", tfidf_vectorizer],
    ["bigram", bigram_vectorizer],
    ["1-2-gram", one2twoGram_vectorizer]
]

# create, train and test naive baysian spam classifier for every vectorization approach
for vectorizer in vectorizers:

    # vectorize train and test data
    x_train_vectorized = vectorizer[1].fit_transform(x_train.values)
    x_test_vectorized = vectorizer[1].transform(x_test.values)

    # set up naive baysian model
    nb = MultinomialNB()

    # train naive baysian model
    nb.fit(x_train_vectorized, y_train)

    # test model
    pred = nb.predict(x_test_vectorized)

    # evaluate test
    score = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test.values, pred, labels=['spam', 'ham'])
    print("vectorization approach:", vectorizer[0])
    print("accuracy:", score)
    print("confusion matrix (spam-ham):", cm)
    print("\n")

# spam classifier based on a feed forward neural network
# follow this tutorial: https://towardsdatascience.com/feed-forward-neural-networks-how-to-successfully-build-them-in-python-74503409d99a
# use sklearn's MLPClassifier to build feedforward neural net: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
# example of how to use sklearn MLPClassifier object: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
for vectorizer in vectorizers:
    # vectorize train and test data
    x_train_vectorized = vectorizer[1].fit_transform(x_train.values)
    x_test_vectorized = vectorizer[1].transform(x_test.values)

    # set up feedforward neural net classifier
    mlp = MLPClassifier(random_state=1, max_iter=300)
    
    # train neural net
    mlp.fit(x_train_vectorized, y_train)

    # test neural net
    pred = mlp.predict(x_test_vectorized)

    # evaluate test
    score = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test.values, pred, labels=['spam', 'ham'])
    print("vectorization approach:", vectorizer[0])
    print("accuracy:", score)
    print("confusion matrix (spam-ham):", cm)
    print("\n")
