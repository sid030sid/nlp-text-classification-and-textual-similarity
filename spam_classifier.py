import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# import data
corpus = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])

# task 2: pre-processing:
# remove stop words, punctuations and white spaces as well as normalise tokens by making all tokens lower case

# create train and test set
x_train, x_test, y_train, y_test = train_test_split(corpus.message, corpus.label, test_size=0.2, random_state=53)

# set up vectorizers
## ... using bag of words --> requires following pre-processing according to lecture: Lowercasing, Punctuation removal, Word tokenization, Stop word removal
bow_vectorizer = CountVectorizer(stop_words='english') # init CountVectorizer object with set up to erase stopwords

## ... using tfidf --> requires following pre-processing according to lecture: Lowercasing, Punctuation removal, Word tokenization, Stop word removal
tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english") # add eventually param: max_df=0.7 --> this erases additional stop words or words that occure to often accross all documents

## ... using bag of N-Gram (n=2) --> requires following pre-processing according to lecture:  Lowercasing, Punctuation removal, Word tokenization
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))

## ... using bag of N-Gram (n=1-2) 
one2twoGram_vectorizer = CountVectorizer(ngram_range=(1, 2))

## summarize vectorizers
vectorizers = [
    ["bow", bow_vectorizer], 
    ["tfidf", tfidf_vectorizer],
    ["bigram", bigram_vectorizer],
    ["1-2-gram", one2twoGram_vectorizer]
]

# create naive baysian spam classifier for every vectorization approach
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