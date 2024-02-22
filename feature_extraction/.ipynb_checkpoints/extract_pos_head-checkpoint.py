import numpy as np
import pandas as pd

def extract_UPOS_of_head(sentence):
    """
    Extract UPOS of head of each word in a sentence.
    
    Return:
        A list of UPOS of head of each word.

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
