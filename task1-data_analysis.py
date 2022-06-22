import numpy as np

#Import data set: SMSSpamCollection

corpus = np.genfromtxt(fname='./data/SMSSpamCollection.txt')


#Analysis of data set: SMSSpamCollection
##Graph comparing length of spam and normal messages.

##Number of unique words in each category (spam and normal messages) and also in the whole dataset

###consider this guide:
# https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools
# or this one: https://towardsdatascience.com/nlp-part-3-exploratory-data-analysis-of-text-data-1caa8ab3f79d
# look at ISIS course. what is suggested?

###eventually do:
###Number of spelling mistakes in each category (spam and normal messages) and also in the whole dataset
###
