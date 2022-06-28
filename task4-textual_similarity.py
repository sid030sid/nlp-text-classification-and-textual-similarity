import numpy as np
import pandas as pd
import spacy

# Create the nlp object
nlp = spacy.load("en_core_web_lg")

# Import data set: SMSSpamCollection
data = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])
corpus = pd.DataFrame(data.label)
corpus["doc"] = data.message.apply(lambda doc : nlp(doc)) # includes pre-proccessing step: tokenization and removal of white spaces

# Pre-processing:  
# --> to do: should be different as this is a new case than before with spam filtering
## remove noises: punctuations, white spaces, stop words
## normalisation: lower case and lemmatization

# choose 15 random spam messages
sample = corpus[corpus.label == "spam"].sample(15)
print(sample.head())

# compute semantic textual similarity in form of cosine similarity by using the average of word vectors as a distributional semantics approach in sentence level
## spacy's similarity function uses average of word vectors when it comes to comparing whole docs (source: https://spacy.io/usage/linguistic-features#similarity-expectations)
## Per default spacy's similarity function for Document Objects uses cosine similarity to determine similarity in per centage. (source: https://spacy.io/api/doc#similarity)
## to do: erase stop words and do all the pre-processing before making similarity check --> if essentiel
similarity_analysis = pd.DataFrame(sample.doc)
similarity_analysis["cosine_similarity_score"] = similarity_analysis.doc.apply(lambda doc : [doc.similarity(doc2) for doc2 in similarity_analysis.doc]) 

'''
i = 0
for doc in sample["doc"]:
    i2 = 0
    for doc2 in sample["doc"]:

        similarity_analysis = pd.concat(
            [
                similarity_analysis,
                {
                    "sample_id#1" : i, 
                    "sample_id#2" : i2, 
                    "average_word_vector#1" : "TBA", 
                    "average_word_vector#2": "TBA", 
                    "cosine_similarity_score": doc.similarity(doc2)
                }
            ]
            ,
            ignore_index=True,
            axis=0
        )
        i2 = i2 + 1
    i = i + 1
'''

###print(sample["doc"].iloc[0].similarity(sample["doc"].iloc[1]))
print(similarity_analysis.head(5))