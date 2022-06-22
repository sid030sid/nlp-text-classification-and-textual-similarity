import pandas as pd
from matplotlib import pyplot
from spacy.lang.en import English   

# Create the nlp object
nlp = English()

#Import data set: SMSSpamCollection
data = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])
print(data.head(5))

#Analysis of data set: SMSSpamCollection
##Graph comparing length of spam and normal messages
stats = data
stats["length"] = stats.apply(lambda row : len(row["message"]), axis=1)
stats["word_count_based_on_whitespaces"] = stats.apply(lambda row : len(row["message"].split()), axis=1)
###to do: stats["count_special_symbols"] = stats.apply(lambda)
for message in data["message"]:
    doc = nlp(message)
    count_numbers_in_message = 0
    for token in doc:
        # Check if the token resembles a number
        if token.like_num:
            count_numbers_in_message = count_numbers_in_message + 1
    print("number of numerals: ", count_numbers_in_message, "; message: ", message)



pyplot.hist(stats.filter(like="spam").select("length"), alpha=0.5, label='spam')
pyplot.hist(stats.filter(like="ham")["length"], alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.show()

##Number of unique words in each category (spam and normal messages) and also in the whole dataset

###consider this guide:
# https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools
# or this one: https://towardsdatascience.com/nlp-part-3-exploratory-data-analysis-of-text-data-1caa8ab3f79d
# look at ISIS course. what is suggested?

###eventually do:
###Number of spelling mistakes in each category (spam and normal messages) and also in the whole dataset
###
