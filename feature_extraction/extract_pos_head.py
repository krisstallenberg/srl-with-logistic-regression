import numpy as np
import pandas as pd

def extract_UPOS_of_head(sentence):
    """
    Extract the token position related to predicate.
    
    Return:
        Before: if token is before predicate in sentence
        After: if token is after predicate in sentence
        _: if token is predicate itself

    sentence (dict): The sentence object
    """

    features = []
    for word in sentence:
        # takes UPOS of words' head, if the word is root, it gives "root"
        head_ID = word['head']
        if int(head_ID) == 0:
            head_pos = 'root'
        else:    
            head_pos = sentence[int(head_ID)-1]['upos']
        features.append(head_pos)

    return features
