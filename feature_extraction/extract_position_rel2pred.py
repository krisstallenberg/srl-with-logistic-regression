import numpy as np
import pandas as pd

def extract_word_position_related_to_predicate(sentence):
    """
    Extract the token position related to predicate.
    
    Return:
        Before: if token is before predicate in sentence
        After: if token is after predicate in sentence
        _: if token is predicate itself

    sentence (dict): The sentence object
    """
    # flag that shows word is before or after the predicate
    is_before = True
    features = []
    for word in sentence:
        # check if word is predicate itself append '_' to features list and reverse the is_before flag.
        if word['predicate'] != '_': 
            features.append('_')
            is_before = False
            continue
        # for other words based in is_before flag value append 'Before' or 'After' to features list 
        features.append('Before' if is_before else 'After')

    return features