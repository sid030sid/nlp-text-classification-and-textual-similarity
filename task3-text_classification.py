# follow this guide: https://www.kdnuggets.com/2020/07/spam-filter-python-naive-bayes-scratch.html which uses th same data set
import numpy as np
import pandas as pd
import spacy

# Create the nlp object
nlp = spacy.load("en_core_web_lg")

# Import data set: SMSSpamCollection
data = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])
corpus = pd.DataFrame(data.label)
corpus["docs"] = data.message.apply(lambda doc : nlp(doc)) # includes pre-proccessing step: tokenization and removal of white spaces

# Pre-processing to dooo
## note: consider that naive baysian algorithm considers total amount of words in spam and non-spam messages as well as how often a certain word occures in spam or non-spam messages
## remove stop words
## remove punctuations 



# Get training and test data (with ratio: 80-20) 
# note: Selection is done randomly. However, we ensure to have approx. 15% of spam messages in our test set
def split_data(df) :
    ## randomize
    df = df.sample(frac=1, random_state=1)

    ## Calculate index for split
    training_test_index = round(len(df) * 0.8)

    ## Split into training and test sets
    training_set = df[:training_test_index].reset_index(drop=True)
    test_set = df[training_test_index:].reset_index(drop=True)

    ## Ensure at least 10-15% of spam messages in test set (to keep ratio like in corpus)
    if not (len(test_set[test_set.label == "spam"]) / len(test_set) <= 0.15 and len(test_set[test_set.label == "spam"]) / len(test_set) >= 0.1) : 
        split_data(df)

    return(training_set, test_set)


## split corpus
training_set, test_set = split_data(corpus)

print(training_set.head(10))
print(test_set.head(10))
print("shape train: ", training_set.shape(), "shape test: ", test_set.shape())

# model
## compute constants
prob_spam = len(training_set[training_set.label == "spam"]) / len(training_set)
prob_ham = len(training_set[training_set.label == "ham"]) / len(training_set)
number_of_tokens_spam = sum([doc.len for doc in training_set[training_set.label == "spam"].docs])
number_of_tokens_ham = sum([doc.len for doc in training_set[training_set.label == "ham"].docs])

## input doc of type nlp object (ideally pre-processed)
### alternative to find occurances of token in all docs:
'''
from spacy.matcher import Matcher
pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
matcher = Matcher(nlp.vocab)
matcher.add("HelloWorld", [pattern])
matches = matcher(doc)
'''

def naive_baysian_spam_filter(doc):

    # calculate conditional probability for every token in to be classified doc
    prob_doc_spam = 1
    prob_doc_ham = 1
    for token in doc:
        occurances_ham = 0
        occurances_spam = 0

        for spam_doc in training_set[training_set.label == "spam"].docs:
            for spam_token in spam_doc:
                if spam_token == token:
                    occurances_spam = occurances_spam + 1

        for ham_doc in training_set[training_set.label == "ham"].docs:
            for ham_token in ham_doc:
                if ham_token == token:
                    occurances_ham = occurances_ham + 1
        prob_doc_spam = prob_doc_spam * occurances_spam / number_of_tokens_spam
        prob_doc_ham = prob_doc_ham * occurances_ham / number_of_tokens_ham

    # calculate final conditioanl probabilities
    prob_spam_doc = prob_spam * prob_doc_spam
    prob_ham_doc = prob_ham * prob_doc_ham

    # make classification decision
    if prob_spam_doc > prob_ham_doc:
        return "spam"
    else:
        return "ham"

# test model
test_set["classification_decision"] = test_set.docs[lambda doc : naive_baysian_spam_filter(doc)]

# assess performance