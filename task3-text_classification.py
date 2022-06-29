# follow this guide: https://www.kdnuggets.com/2020/07/spam-filter-python-naive-bayes-scratch.html which uses th same data set
import numpy as np
import pandas as pd
import spacy

# Create the nlp object
nlp = spacy.load("en_core_web_lg")

# Import data set: SMSSpamCollection
data = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])
corpus = pd.DataFrame(data.label)
corpus["docs"] = data.message.apply(lambda doc : nlp(doc.lower())) # includes pre-proccessing step: tokenization, removal of white spaces and normalizing (lower case)

# Pre-processing to dooo
## note: consider that naive baysian algorithm considers total amount of words in spam and non-spam messages as well as how often a certain word occures in spam or non-spam messages

## remove stop words, punctuations and white spaces as well as normalise tokens by making all tokens lower case
corpus["processed_docs"] = corpus.docs.apply(lambda doc : [token for token in doc if not token.is_stop and not token.is_punct and not token.is_space])

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
print("training data - number of spam messages: ", len(training_set[training_set.label == "spam"]), "number of ham messages", len(training_set[training_set.label == "ham"]))
print("test data - number of spam messages: ", len(test_set[test_set.label == "spam"]), "number of ham messages", len(test_set[test_set.label == "ham"]))

# model
## compute constants
prob_spam = len(training_set[training_set.label == "spam"]) / len(training_set)
prob_ham = len(training_set[training_set.label == "ham"]) / len(training_set)
number_of_tokens_spam = sum([len(doc) for doc in training_set[training_set.label == "spam"].docs])
number_of_tokens_ham = sum([len(doc) for doc in training_set[training_set.label == "ham"].docs])
'''
print("prob_spam:", prob_spam)
print("prob_ham:", prob_ham)
print("number_of_tokens_spam:", number_of_tokens_spam)
print("number_of_tokens_ham:", number_of_tokens_ham)
'''

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

        for spam_doc in training_set[training_set.label == "spam"].processed_docs:
            for spam_token in spam_doc:
                if spam_token.text == token.text:
                    occurances_spam = occurances_spam + 1

        for ham_doc in training_set[training_set.label == "ham"].processed_docs:
            for ham_token in ham_doc:
                if ham_token.text == token.text:
                    occurances_ham = occurances_ham + 1

        prob_token_spam = occurances_spam / number_of_tokens_spam
        prob_token_ham = occurances_ham / number_of_tokens_ham

        prob_doc_spam = prob_doc_spam * prob_token_spam 
        prob_doc_ham = prob_doc_ham * prob_token_ham 

    # calculate final conditioanl probabilities
    prob_spam_doc = prob_spam * prob_doc_spam
    prob_ham_doc = prob_ham * prob_doc_ham

    # make classification decision
    if prob_spam_doc >= prob_ham_doc:
        return "spam"
    else:
        return "ham"

# test function
'''
print(test_set.processed_docs[0])
print("spam filter result:", naive_baysian_spam_filter(test_set.processed_docs[0]), "actual label:", test_set.label[0])

for word in ["later", "guess", "needa", "mcat", "study"]:
    for i, doc in enumerate(training_set[training_set.label == "spam"].processed_docs):
        for token in doc:
            if token.text == "later":
                print(word, "should not exist. It is found in training set doc", i)
'''
# test model
test_set["classification_decision"] = test_set.processed_docs.apply(lambda doc : naive_baysian_spam_filter(doc))
print(test_set.head(10))

# assess performance (F1 score)
tp = 0 #positive = spam
fp = 0 
tn = 0 #negative = ham
fn = 0
for index, variable in test_set.iterrows():
    # correct classification
    if variable["classification_decision"] ==  variable["label"]:
        if variable["classification_decision"] == "spam":
            tp = tp + 1
        else:
            tn = tn +1
    else: 
        if variable.classification_decision == "spam":
            fp = fp + 1
        else:
            fn = fn +1
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print("tp: ", tp,", fp:", fp,", tn:", tn,", fn:", fn)
print("F1:", f1) # 1 is perfect and 0 is very bad
        

# compare the naive baysian model'S performance for different vectorization methods
## to do
