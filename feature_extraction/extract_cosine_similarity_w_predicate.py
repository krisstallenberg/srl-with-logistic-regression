from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm

def extract_cosine_similarity_w_predicate(sentence, model):
    """
    Extract the cosine similarity between the predicate and every word.

    Returns a list of cosine similarity values between the predicate and every word in the sentence.

    sentence (list): The sentence list of objects.
    model: the loaded pretrained word embedding model.
    """
    features = []

    # Extract the predicate's lemma
    predicate_lemma = [word['lemma'] for word in sentence if word['predicate'] != '_'][0]
    
    # If the predicate's lemma is in the model, extract its vector, else create a vector of zeros
    if predicate_lemma in model:
        predicate_lemma_vector = model[predicate_lemma]
    else:
        predicate_lemma_vector = [0.0] * 200
        
    for word in sentence:
        # If the word's lemma is in the model, extract its vector, else create a vector of zeros
        if word['lemma'] in model:
            word_lemma_vector = model[word['lemma']]
        else:
            word_lemma_vector = [0.0] * 200

        # Calculate the cosine similarity between the predicate and the word, and append it to features
        features.append(cosine_similarity([predicate_lemma_vector], [word_lemma_vector])[0][0])

    return features
