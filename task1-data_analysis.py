import numpy as np
import pandas as pd
from matplotlib import pyplot
import spacy
from collections import Counter
import nltk
import dataframe_image as dfi

# Create the nlp object
nlp = spacy.load("en_core_web_lg")

# Import data set: SMSSpamCollection
data = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])

# Analysis of data set: SMSSpamCollection
## Analysis of column: message
stats = pd.DataFrame(data)

### Length of message (=number of characters)
stats["length"] = stats.apply(lambda row : len(row["message"]), axis=1)

### Number of numbers per message
stats["count_numbers"] = stats.apply(lambda row : sum(c.isdigit() for c in row["message"]), axis=1)

### Number of letters numbers per message
stats["count_letters"] = stats.apply(lambda row : sum(c.isalpha() for c in row["message"]), axis=1)

### Number of whitespaces per message
stats["count_spaces"] = stats.apply(lambda row : sum(c.isspace() for c in row["message"]), axis=1)

### Number of dots per message
stats["count_dots"] = stats.apply(lambda row : sum(c == "." for c in row["message"]), axis=1)

### Number of other literals per message
stats["count_other_literals"] = stats.apply(lambda row : (row["length"] - row["count_numbers"] - row["count_letters"] - row["count_spaces"] - row["count_dots"]), axis=1)

### Number of words per messaged (based on seperation by whitespacs)
#to do: think about how this would be better if it is done with spacy --> does not matter how as long as you specifically mention the implications of chosing this approach in the report
# using this apporach means that words are counted which are not words e.g. "..." or "word..."
stats["word_count_based_on_whitespaces"] = stats.apply(lambda row : len(row["message"].split()), axis=1)

### Average length of words per message
#to do: think about how this would be better if it is done with spacy
stats["average_word_length"] = stats.apply(lambda row : sum(len(word) for word in row["message"].split()) / row["word_count_based_on_whitespaces"], axis=1)



## Visualisation of text
### Graph comparing length of spam and normal messages (length = number of literals)
pyplot.subplot(3,1,1)
pyplot.hist(stats[stats.label == "spam"].length, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].length, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of length of message for spam or ham (non-spam)')  
pyplot.xlabel('Length in number of literals')
pyplot.ylabel('Number of messages')

### Graph comparing the amount of words in spam and non-spam
pyplot.subplot(3,1,2)
pyplot.hist(stats[stats.label == "spam"].word_count_based_on_whitespaces, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].word_count_based_on_whitespaces, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of amount of words in one message for spam or ham (non-spam)')  
pyplot.xlabel('Amount of words')
pyplot.ylabel('Number of messages')

### Graph comparing the average word length of any message for spam and non-spam
pyplot.subplot(3,1,3)
pyplot.hist(stats[stats.label == "spam"].average_word_length, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].average_word_length, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of average word length in one message for spam or ham (non-spam)')  
pyplot.xlabel('Average word length')
pyplot.ylabel('Number of messages')

### print visualisations
pyplot.show()



## Visualisation of analysis of literals
### Graph comparing the number of numbers in spam and ham messages
pyplot.hist(stats[stats.label == "spam"].count_numbers, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].count_numbers, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of amount of numbers in one message for spam or ham (non-spam)')  
pyplot.xlabel('Amount of numbers')
pyplot.ylabel('Number of messages')
pyplot.show()

### Graph comparing the number of dots in spam and ham messages
pyplot.hist(stats[stats.label == "spam"].count_dots, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].count_dots, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of amount of dots in one message for spam or ham (non-spam)')  
pyplot.xlabel('Amount of dots')
pyplot.ylabel('Number of messages')
pyplot.show()

### Graph comparing the number of whitespaces in spam and ham messages
pyplot.hist(stats[stats.label == "spam"].count_spaces, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].count_spaces, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of amount of whitespaces in one message for spam or ham (non-spam)')  
pyplot.xlabel('Amount of whitespaces')
pyplot.ylabel('Number of messages')
pyplot.show()

### Graph comparing the number of special letters [special = not a letter, number, whitespace or dot] in spam and ham messages
pyplot.hist(stats[stats.label == "spam"].count_other_literals, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].count_other_literals, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of amount of special literals in one message for spam or ham (non-spam)')  
pyplot.xlabel('Amount of special literals')
pyplot.ylabel('Number of messages')
pyplot.show()



#Analysis via spacy package
corpus = pd.DataFrame(data.label)
corpus["doc"] = data.message.apply(lambda doc : nlp(doc))  #creating a nlp object includes first pre-processing: tokenization

"""
not necessary
corpus["words"] = corpus.doc.apply(lambda doc : ([token for token in doc if not token.is_stop and not token.is_punct]))
corpus["stopwords"] = corpus.doc.apply(lambda doc : ([token for token in doc if token.is_stop]))
corpus["punctuations"] = corpus.doc.apply(lambda doc : ([token for token in doc if token.is_punct]))
"""

## Number of unique words in each category (spam and normal messages) and also in the whole dataset
### for spam messages
spam_word_freq = Counter([token.text for doc in corpus[corpus.label == "spam"].doc for token in doc if token.is_alpha and not token.is_stop])
spam_unique_words = [item for item in spam_word_freq.most_common() if item[1] == 1]
print("Number of unique words in spam messages: ", len(spam_unique_words))

### for ham messages
ham_word_freq = Counter([token.text for doc in corpus[corpus.label == "ham"].doc for token in doc if token.is_alpha and not token.is_stop])
ham_unique_words = [item for item in ham_word_freq.most_common() if item[1] == 1]
print("Number of unique words in non-spam messages: ", len(ham_unique_words))

## Top 10 of most used words in each category (spam and normal messages)
### note: no normalisation has been done, thus "FREE" and "free" are listed seperatly. This highlights the necessity for normalisation by making all tokens in documnets lower case!
### for spam messages
print("Top 10 most used words in spam messages: ", spam_word_freq.most_common(10))
pyplot.bar([item[0] for item in spam_word_freq.most_common(10)], [item[1] for item in spam_word_freq.most_common(10)])
pyplot.title('Top 10 most used words in spam messages')  
pyplot.show()

### for ham messages
print("Top 10 most used words in non-spam messages: ", ham_word_freq.most_common(10))
pyplot.bar([item[0] for item in ham_word_freq.most_common(10)], [item[1] for item in ham_word_freq.most_common(10)])
pyplot.title('Top 10 most used words in non-spam messages')  
pyplot.show()



## word cloud for each category (= visualisation of frequency of words per category)
### to do: eventually



## Stop word analysis:
### add number of stops words to df: stats
stats["count_stop_words"] = corpus.apply(lambda row : sum([1 for token in row["doc"] if token.is_stop]), axis=1)

### Graph comparing the number of stop words in spam and ham messages
pyplot.hist(stats[stats.label == "spam"].count_stop_words, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].count_stop_words, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of amount of stop words in one message for spam or ham (non-spam)')  
pyplot.xlabel('Amount of stop words')
pyplot.ylabel('Number of messages')
pyplot.show()

### Number of stopwords in each category (spam and normal messages) and also in the whole dataset
#### for spam messages
spam_stopwords = Counter([token.text for doc in corpus[corpus.label == "spam"].doc for token in doc if token.is_stop])
print("Number of words in spam messages: ", spam_stopwords.total())

#### for ham messages
ham_stopwords = Counter([token.text for doc in corpus[corpus.label == "ham"].doc for token in doc if token.is_stop])
print("Number of words in non-spam messages: ", ham_stopwords.total())

### Top 10 of most used stop words
#### for spam messages
print("Top 10 most used stopwords in spam messages: ", spam_stopwords.most_common(10))
pyplot.bar([item[0] for item in spam_stopwords.most_common(10)], [item[1] for item in spam_stopwords.most_common(10)])
pyplot.title('Top 10 most used stop words in spam messages')
pyplot.show()

#### for ham messages
print("Top 10 most used stopwords in non-spam messages: ", ham_stopwords.most_common(10))
pyplot.bar([item[0] for item in ham_stopwords.most_common(10)], [item[1] for item in ham_stopwords.most_common(10)])
pyplot.title('Top 10 most used stop words in non-spam messages')
pyplot.show()

### to do eventually: find out number of nouns, pos etc. for each category



## content analysis
### to do eventually: find out topic


## summary of text analysis
summary1 = stats.groupby("label").agg({
    "length" : ["min", "mean", "median", "max"],
    "word_count_based_on_whitespaces" : ["sum", "min", "mean", "median", "max"],
    "count_stop_words" : ["sum", "min", "mean", "median", "max"],
    "average_word_length" : ["min", "mean", "median", "max"]
})
summary1["count_unique_words"] = [len(ham_unique_words), len(spam_unique_words)]
dfi.export(summary1, "documentation/tables_as_image/summary_text_analysis.png")

## Summary of analysis of literals
summary2 = stats.groupby("label").agg({
    "count_letters" : ["min", "mean", "median", "max"],
    "count_numbers" : ["min", "mean", "median", "max"],
    "count_dots" : ["min", "mean", "median", "max"],
    "count_spaces" : ["min", "mean", "median", "max"],
    "count_other_literals" : ["min", "mean", "median", "max"]
})
dfi.export(summary2, "documentation/tables_as_image/summary_literal_analysis.png")

###all upper to dos are based on the given task or this guide:https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools
# or this one: https://towardsdatascience.com/nlp-part-3-exploratory-data-analysis-of-text-data-1caa8ab3f79d
# look at ISIS course. what is suggested?

###eventually do:
###Number of spelling mistakes in each category (spam and normal messages) and also in the whole dataset
###