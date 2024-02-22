import numpy as np
import pandas as pd

def find_dependency_path(token_id, sentence,pred_id ):
    """
    Extract the depndency path of each token to predicate.
    
    Return:
        List: a dependency path from token_id to pred_id.
        Using padding, list has always len of 20. Because MAX dependency path in all used datasets is 17.

    sentence (dict): The sentence object
    token_id : The id of the word in the sentence
    pred_id : The id of the predicate in the sentence
    """
    dep_path = ['']*20
    current_id = token_id
    idx=0
    # Starts from token_id and each time 'form' and 'XPOS' of that token is added to dependency path list.
    # Then, token becomes head of the current token. It continues until head is the predicate.
    # list with len of 20 with empty strings is created, then from beginning of list empty strings are replaced with the path.
    while current_id != 0:
        if int(current_id)-1 == 0:
            break
        token = sentence[int(current_id)-1]

        if token is None:
            break

        dep_path[idx] = (f"{token['xpos']}")
        idx +=1
        current_id = int(token['head'])

    return dep_path

def extract_dependency_path(sentence):
    """
    Extract the depndency path of each token to predicate.
    
    Return:
        A list of dependency paths.

    sentence (dict): The sentence object
    """
    features = []
    # Find id of predicate in a sentence.
        
    for i, word in enumerate(sentence):
        if word['predicate'] != '_':
            pred_id = i
    # For each word in the sentence, if it is not root or prediate, finds its dependency path to predicate.
    # If it is root or predicate, its dependency path is a list of empty strings with len of 20.
    for i, word in enumerate(sentence):
        if word['form'] != sentence[int(pred_id)]['form']:
            dep_path = find_dependency_path(word['id'], sentence,pred_id)
            if dep_path:
                features.append(dep_path)

            # If the current word is the root, its dependency path is a list of empty strings with len of 20
            else:
                features.append(['']*20)

        # If the current word is the predicate, its dependency path is a list of empty strings with len of 20
        else:
            features.append(['']*20)

    return features
