# read before and follow: https://www.kdnuggets.com/2017/02/natural-language-processing-key-terms-explained.html
# to do: find out automatic pre-processing steps done by creating spacy's nlp object

# Tokenization and segmentation

# Noise removal: only keep tokens containing alpha, numbers and punctuations like (!, ? etc)
## note lecturer slide: 

# Noise removal is about removing characters, digits and pieces of text that can interfere with your text analysis. 
# It’s highly domain dependent
##to do: think about sensefullness of this measurement? only keep punctuations which can be identified for spam: e.g. !, ? ... what else? task 1 revelas that number of dots can be decisive for spam or not spam?

# Normalization: 
## remove white spaces --> done by spacy automatically?

## remove stop words

## remove single dots 

## make all tokens lower case

## lemmatization: to do - does it make sense?
### Note lecture slide: The goal of both stemming and lemmatization is to reduce inflectional 
# forms of a word to a common base form
# • Stemming is the process of eliminating affixes (suffixed, prefixes, infixes, circumfixes) from a 
# word in order to obtain a word stem
# • Lemmatization is related to stemming, differing in that lemmatization is able to capture canonical forms based on a word’s lemma
