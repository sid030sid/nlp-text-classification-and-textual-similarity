# nlp-text-classification-and-textual-similarity

## Getting the project runnning
1. create virtual environment by entering in terminal: py -3 -m venv .venv
2. activate newly created environment by entering in terminal: .venv\scripts\activate
3. download packages by entering in terminal witch active virtual environment: python -m pip install package_name
4. run phyton file by doing right click on respective file and selecting "Run Phyton file in terminal" or alternatively by entering in terminal witch active virtual environment: python file_name.py

## TASKS

### Task 1: Extract insights from data (15%)
For this task you should extract some insights (i.e., some statistics and graphs) from the provided data.
Here are some ideas:
- It could be a graph compared the length of spam and normal messages.
- Number of unique words in each category and also in the whole dataset
- …
Please don’t limit yourself to these two examples and try to summarize the data with numbers and
graphs. Please highlight some of the most important findings in your report.

### Task 2: Pre-processing (20%)
In this task, first, apply all the necessary pre-processing steps that you think they would help to better
prepare your data for the next steps. You don’t have to apply all the pre-processing tasks which are
covered in the course. Regarding the report, you should briefly mention it in your report that why you
decided to apply the chosen pre-processing steps (and why not the others).

### Task 3: Text classification (50%)
In this task you should do the following sub-tasks:
- Split data into train and test sets. Use 20% of data as the test set.
- Train a naïve Bayes model on the training part and test it, using the test set.
o Compare the impact of different vectorization models (e.g., count vectorizer, TF-IDF
and …) on the final performance of your naïve Bayes model.
- Train a feed forward neural network model and report its performance (F1 score) on detecting
spam messages on test data.
o Again, compare the impact of different vectorization approaches on the final
performance of your model.
Please report all the achieved results with either models in your report document. Moreover, describe
the hyper-parameters of your neural network model in the report.

### Task 4: Textual similarity (15%)
In this task, you should choose 15 random spam messages, and compute semantic textual similarity
between them. Please use the average of word vectors as a distributional semantics approach in
sentence level to measure similarity between messages. Please report the cosine similarity between
randomly selected sentences in your report.

### Bonus Task: Textual similarity (+20%)
As the bonus task, you should fine-tune two pre-trained models on the provided spam detection
dataset. You can choose between the wide range of pre-trained models (e.g., BERT, RoBERTa, and so
on). It is very important to be careful about the hyper-parameters and fine-tune the models as
accurate as possible without too much overwriting of the current weights. Please describe your
selected models and the training process in details in the report.