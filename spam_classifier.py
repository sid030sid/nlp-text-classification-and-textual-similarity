# to do: 
#task 3: do cross validation for proper performance comparison: https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
#task3: how does cross validation reveal overfitting? how to interpret cross validation result?
#task 4: in a clean way and in its on py file
#task 1 + write report
# bonus task

# to do for report:
# 1. mention that multionominalNB was used. why? one of two common classifiers in text classification (soruce: https://scikit-learn.org/stable/modules/naive_bayes.html)
# 2. highlight differences between mutlinominalNB as taught in lecture 
# --> multinomoinalNB uses laplace smoothing which "accounts for features not present in the learning samples and prevents zero probabilities" (soruce: https://scikit-learn.org/stable/modules/naive_bayes.html). the NB according to lecture does not account for this. for the case that one token does no appear in whole training set for one classification category the whole document's probability to be in the respective classification category is zero
# 3.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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

# hyper parameters of used feed forward neural net which is based on sklearn's MLPClassifier object (source: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
# 1. activation function --> Relu was chosen as it is usally the best one according to the lecture. sklearn's MLPClassifier object user per default Relu.
# 2. loss function --> from doc: "This model optimizes the log-loss function using LBFGS or stochastic gradient descent." Thus, the feed forward neural net uses the default log-loss function which is also recomended by the lecture.
# 3. number of hidden layers --> default value = n_layers - 2 = 3 - 2 = 1. The used neural net uses one hidden layer as "or many practical problems, there is no reason to use any more than one hidden layer." (soruce: Introduction to Neural Networks for Java (second edition) by Jeff Heaton [near table 5.1.] --> reference:https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)
# 4. number of neurons per hidden layer --> default value is used which is 100 --> as it is a matter of try and error to find an optimal number of neurons per hidden layer (according to Jeff Heaton). We can go with the first try as it outputd promising accuracy results

# = 2/3 * input_layer_size + output_layer_size = 2/3 * ... to do ... + 1 (input_layer_size = number of input variables in the data being processed [soruce: https://towardsdatascience.com/beginners-ask-how-many-hidden-layers-neurons-to-use-in-artificial-neural-networks-51466afa0d3e])
# Jeff Heaton mentions three rule-of-thumb methods for chosing the number of neurons per hidden layer. 
# For the used neural net the second method was followed, namely "The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer." 
# Nonetheless, "the selection of an architecture for your neural network will come down to trial and error". (according to Jeff Heaton)

# 5. design of output layer --> sklearn's MLPClassifier default is used which is 1. This can be found out by calling the n_outputs_ attribute of the initiate MLPClassifier object.
# 6. all other hyper param are set to default of sklearn's MLPClassifier
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
    accurcy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test.values, pred, labels=['spam', 'ham'])
    print("vectorization approach:", vectorizer[0])
    print(
        "hyper parameters:\nnumber of output neurons:", mlp.n_outputs_,
        "\nnumber of hidden layers:", mlp.n_layers_,
        "\nused activation function:", mlp.out_activation_,
        "\nclasses:", mlp.classes_
    )
    print("accuracy:", score)
    print("confusion matrix (spam-ham):", cm)
    print("\n")
