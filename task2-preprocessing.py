# read before and follow: https://www.kdnuggets.com/2017/02/natural-language-processing-key-terms-explained.html
import numpy as np
import pandas as pd
import spacy

# Create the nlp object
nlp = spacy.load("en_core_web_lg")

# Import data set: SMSSpamCollection
data = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])
corpus = pd.DataFrame(data.label)



# Tokenization and segmentation
## note: using spacy's nlp function and transforming every message of df: data into a nlp object aka document made of tokens, comes with some automatic pre-procssing steps.
## These automatic pre-processing steps are: removal of whitespaces
corpus["doc"] = data.message.apply(lambda doc : nlp(doc))
print(corpus.head(5))
print(corpus.doc[0])



# Noise removal: only keep tokens containing alpha, numbers and punctuations like (!, ? etc)
## note lecturer slide: 
# Noise removal is about removing characters, digits and pieces of text that can interfere with your text analysis. 
# It’s highly domain dependent
##to do: think about sensefullness of this measurement? only keep punctuations which can be identified for spam: e.g. !, ? ... what else? task 1 revelas that number of dots can be decisive for spam or not spam?
## remove single dots --> usage of those indicates spam or not-spam

## remove ",", ":", ";" --> does it make sense?



# Normalization: 
## remove white spaces --> done by spacy automatically at point of creation of NLP object

## remove stop words

## make all tokens lower case

## spelling correction: --> only do if not too time consuming: https://www.naukri.com/learning/articles/text-pre-processing-for-spam-filtering/

## lemmatization: to do - does it make sense?
### Note lecture slide: The goal of both stemming and lemmatization is to reduce inflectional 
# forms of a word to a common base form
# • Stemming is the process of eliminating affixes (suffixed, prefixes, infixes, circumfixes) from a 
# word in order to obtain a word stem
# • Lemmatization is related to stemming, differing in that lemmatization is able to capture canonical forms based on a word’s lemma
