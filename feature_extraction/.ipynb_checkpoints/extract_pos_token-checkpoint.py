def extract_pos_token(sentence):
    """
    Extract the POS tag of each token in the sentence.

    Returns a list of the tokens contained in a sentence.

    sentence (dict): a Python dictionary containing all the columns from CoNLLu file for each token.
    """
    
    #for sentence in data:
    
    pos_token = []
    for dict in sentence: #iterating over the words and their data in the sentence, each stored under a dictionary
        pos_token.append(dict['upos'])

    return pos_token