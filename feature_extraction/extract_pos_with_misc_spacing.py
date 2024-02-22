def pos_misc_feature(sentence):
    """
    Combines POS tag information with 'SpaceAfter=No' for each token in a sentence.

    Returns a list of combined features per token
    """
    features = []
    for dict_info in sentence:  
        pos_tag = dict_info['upos']
        space_info = 'SpaceAfter=No' in dict_info.get('miscellaneous', '') # checks the absence of space ('SpaceAfter=No') in the misc column
        combined_feature = f"{pos_tag}_{'no_space' if space_info else 'space'}" #combines POS tag with space information into a single feature
        
        features.append(combined_feature)
    
    return features