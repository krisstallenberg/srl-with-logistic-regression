# Semantic Role Labeling with Logistic Regression

This repository contains Python code for Semantic Role Labeling (SRL) using the English Universal PropBank.

The code preprocesses the dataset, extracts a set of features and trains a model for argument identification and classification. Finally, it evaluates the model's performance.

## Dependencies

The following libraries are required for running the code:
- pandas
- gensim.downloader
- nltk.corpus
- numpy
- spacy

To run the code, make sure libraries required are installed. They are listed in the requirements.txt file.
They can be installed with the following command:

pip install -r requirements.txt


## Code structure

- main.py: main script for preprocessing, training, and inference.

Scripts provided called in the main function:
- evaluation.py contains functions for evaluating the model's performance.
- feature_extraction/: directory containing scripts for feature extraction.

Directory needed for running the code:
- data/: directory for storing the Universal PropBank dataset.

To execute the code run the following command:

python main.py

Optional argument:

--use-cached: use cached preprocessed files for training and inference.

## Dataset

The task is performed on the Enlgish Universal PropBank v1 dataset in CoNLL-U Plus format available at https://github.com/UniversalPropositions/UP-1.0/tree/master/UP_English-EWT

In order to run successfully run the code, the datasets have to be placed in the abovementioned subdirectory.

## Preprocessing

During preprocessing, the sentences that contain more than one predicate are multiplied according to the number of predicates found.

To perform argument identification and classification for SRL, a predicate is essential. Therefore, the sentences that do not contain a predicate are discarded.

## Feature extraction

A variety of features are extracted to train a logistic regression model, namely:

- The token itself as a onehot vector
- The lemma as a onehot vector
- The universal POS
- The Penn Treebank POS
- The dependency relation to the head of the argument
- The PropBank predicate
- The POS of the head of the argument
- The head of the prepositional phrase
- The NER label of each token
- The position of the token in relation to the predicate
- The POS tag and spacing information of the token
- The embedding of the lemma         
- The cosine similarity of the embedding of the lemma and the embeding of the predicate 

### Word embedding model

The language model used is the 6B tokens 200 dimension GloVe embeddings pretained on Wikipedia data.

The model is loaded with the code provided at https://github.com/piskvorky/gensim-data. As the authors state, when using the Gensim download API, all data is stored in ~/gensim-data home folder.

## Model

The model applied for SRL argument identification and classification is a logistic regression model from the Sklearn library.

When training the model, the preprocessed file is stored under the data subdirectory with pickle which can be loaded again for testing.

## Contributors

Farnaz Bani Fatemi \
Ariana Britez \
Christina Karavida \
Szabolcs Pál \
Kris Stallenberg
              




