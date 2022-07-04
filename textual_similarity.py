import pandas as pd
import spacy
import dataframe_image as dfi

# Create the nlp object
nlp = spacy.load("en_core_web_lg")

# Import data set: SMSSpamCollection
data = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])
corpus = pd.DataFrame(data)
corpus["doc"] = data.message.apply(lambda doc : nlp(doc.lower())) # includes pre-proccessing step: tokenization, removal of white spaces and lower casing

# Pre-processing:  
## remove noises: punctuations, white spaces, stop words
## normalisation: lower case and lemmatization
pre_processed_docs = []
for doc in corpus.doc:
    pre_processed_doc = []
    for token in doc:
        if token.is_alpha and not token.is_stop:
            pre_processed_doc.append(token.lemma_)
    pre_processed_docs.append(pre_processed_doc)
pre_processed_docs = [" ".join(doc) for doc in pre_processed_docs]
corpus["doc"] = pre_processed_docs
corpus["doc"] = corpus.doc.apply(lambda doc : nlp(doc.lower())) # includes pre-proccessing step: tokenization, removal of white spaces and lower casing

# choose 15 random spam messages
sample = corpus[corpus.label == "spam"].sample(15)

# compute semantic textual similarity in form of cosine similarity by using the average of word vectors as a distributional semantics approach in sentence level
similarity_analysis = []
for idx, row in sample.iterrows():
    for idx2, row2 in sample.iterrows():
        if(idx == idx2) : continue
        similarity_analysis.append([row.message, row2.message, row.doc.similarity(row2.doc)])
similarity_analysis = pd.DataFrame(similarity_analysis)
similarity_analysis.columns = ["message", "similar_to", "cosine_similarity_score"]

# summarize analysis: find for every message the most similar messages according to cosine similarity based on averade of word vectors in sentence level
summary_similarity_analysis = similarity_analysis.groupby("message").max()
dfi.export(summary_similarity_analysis, "documentation/tables_as_image/summary_similarity_analysis.png")
summary_similarity_analysis.similar_to