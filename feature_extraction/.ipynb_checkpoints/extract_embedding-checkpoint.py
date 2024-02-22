import numpy as np

def extract_embedding(sentence):
    """
    Extract the embedding of the token at the given index.

    Returns the embedding of the token at the given index.

    sentence (dict): The sentence object
    index (int): The index of the token whose embedding is to be extracted
    """
    features = []
    for word in sentence:
        features.append(np.array([0.0] * 300))

    return features