import numpy as np
import pandas as pd
from matplotlib import pyplot
from spacy.lang.en import English   
from collections import Counter
import nltk

# Create the nlp object
nlp = English()

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

### Number of whtiespaces per message
stats["count_spaces"] = stats.apply(lambda row : sum(c.isspace() for c in row["message"]), axis=1)

### Number of other literals per message
stats["count_other_literals"] = stats.apply(lambda row : (row["length"] - row["count_numbers"] - row["count_letters"] - row["count_spaces"]), axis=1)

### Number of words per messaged (based on seperation by whitespacs)
stats["word_count_based_on_whitespaces"] = stats.apply(lambda row : len(row["message"].split()), axis=1)

### Average length of words per message
stats["average_word_length"] = stats.apply(lambda row : sum(len(word) for word in row["message"].split()) / row["word_count_based_on_whitespaces"], axis=1)



## Summary of analysis
#to do: min, max, mean, median of all numeral columns of stats and group by label (=spam or ham)
summary = pd.DataFrame(stats)



## Visualisation of analysis of column: message
### Graph comparing length of spam and normal messages (length = number of literals)
pyplot.subplot(2,2,1)
pyplot.hist(stats[stats.label == "spam"].length, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].length, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of length of message for spam or ham (non-spam)')  
pyplot.xlabel('Length in number of literals')
pyplot.ylabel('Number of Messages')

### Graph comparing the number of words (found based on seperating by white spaces)
pyplot.subplot(2,2,2)
pyplot.hist(stats[stats.label == "spam"].word_count_based_on_whitespaces, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].word_count_based_on_whitespaces, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of number of words in one message for spam or ham (non-spam)')  
pyplot.xlabel('Number of words (based on seperation by whitespaces)')
pyplot.ylabel('Number of Messages')

### Graph comparing the average word length of any message for spam and non-spam
pyplot.subplot(2,2,3)
pyplot.hist(stats[stats.label == "spam"].average_word_length, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].average_word_length, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of average word length in one message for spam or ham (non-spam)')  
pyplot.xlabel('Average word length')
pyplot.ylabel('Number of Messages')
#pyplot.show()



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
print("Number of words in non-spam messages: ", len(ham_unique_words))

## Top 10 of most used words in each category (spam and normal messages)
### for spam messages
print("Top 10 most used words in spam messages: ", spam_word_freq.most_common(10))
#to do: visualise as bar chart

### for ham messages
print("Top 10 most used words in non-spam messages: ", ham_word_freq.most_common(10))
#to do: visualise



## word cloud for each category (= visualisation of frequency of words per category)
### to do: eventually



## Stop word analysis:
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
#to do: visualise as bar chart

#### for ham messages
print("Top 10 most used stopwords in non-spam messages: ", ham_stopwords.most_common(10))
#to do: visualise as bar chart

###all upper to dos are based on the given task or this guide:https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools
# or this one: https://towardsdatascience.com/nlp-part-3-exploratory-data-analysis-of-text-data-1caa8ab3f79d
# look at ISIS course. what is suggested?

###eventually do:
###Number of spelling mistakes in each category (spam and normal messages) and also in the whole dataset
###
