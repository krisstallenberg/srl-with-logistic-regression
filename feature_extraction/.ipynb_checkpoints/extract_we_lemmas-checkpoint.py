import numpy as np

def extract_embedding_lemma(sentence,model):
    """
    Extract the embedding of the lemma for each token in the sentence.

    Returns a list of embedding vectors of the lemmatized tokens contained in a sentence.

    sentence (dict): a Python dictionary containing all the columns from CoNLLu file for each token.
    model: the loaded pretrained word embedding model (word2vec 200 dim).
    """
    
    #for sentence in data:
    
    embedding_lemma = []
    for dict in sentence: #iterating over the words in the sentence
        if dict['lemma'] in model:
            embedding_lemma.append(model[dict['lemma']])
        else:
            embedding_lemma.append(np.array([0.0] * 300))

    return embedding_lemma