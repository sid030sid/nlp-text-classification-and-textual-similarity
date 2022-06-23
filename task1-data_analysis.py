import pandas as pd
from matplotlib import pyplot
from spacy.lang.en import English   

# Create the nlp object
nlp = English()

#Import data set: SMSSpamCollection
data = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])
print(data.head(5))

#Analysis of data set: SMSSpamCollection
stats = data
stats["length"] = stats.apply(lambda row : len(row["message"]), axis=1)
stats["count_numbers"] = stats.apply(lambda row : sum(c.isdigit() for c in row["message"]), axis=1)
stats["count_letters"] = stats.apply(lambda row : sum(c.isalpha() for c in row["message"]), axis=1)
stats["count_spaces"] = stats.apply(lambda row : sum(c.isspace() for c in row["message"]), axis=1)
stats["count_other_literals"] = stats.apply(lambda row : (row["length"] - row["count_numbers"] - row["count_letters"] - row["count_spaces"]), axis=1)
stats["word_count_based_on_whitespaces"] = stats.apply(lambda row : len(row["message"].split()), axis=1)
#to do stats["average_word_length"] = stats.apply(lambda row : len(row["message"].split()), axis=1)
print(stats.head(5))

##Graph comparing length of spam and normal messages (length = number of literals)
pyplot.hist(stats[stats.label == "spam"].length, alpha=0.5, label='spam')
pyplot.hist(stats[stats.label == "ham"].length, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.title('Histogram of length of message for spam or ham (non-spam)')  
pyplot.xlabel('Length in number of literals')
pyplot.ylabel('Number of Messages')
pyplot.show()

##Graph comparing the number of words (found based on seperating by white spaces)
###to do

#based on spacy
##Number of unique words in each category (spam and normal messages) and also in the whole dataset
### mandatory to do
"""""
for message in data["message"]:
    doc = nlp(message)
    count_numbers_in_message = 0
    for token in doc:
        # Check if the token resembles a number
        if token.like_num:
            count_numbers_in_message = count_numbers_in_message + 1
    print("number of numerals: ", count_numbers_in_message, "; message: ", message)
"""

##Graph comparing the average word length of any message for spam and non-spam
### to do

##number of stopwords in each category (spam and normal messages) and also in the whole dataset
### to do

#top 10 of most often used words in each category (spam and normal messages)
### to do

###all upper to dos are based on the given task or this guide:https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools
# or this one: https://towardsdatascience.com/nlp-part-3-exploratory-data-analysis-of-text-data-1caa8ab3f79d
# look at ISIS course. what is suggested?

###eventually do:
###Number of spelling mistakes in each category (spam and normal messages) and also in the whole dataset
###
