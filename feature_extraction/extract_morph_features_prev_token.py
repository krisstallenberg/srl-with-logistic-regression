def previous_token_morph_features(sentence):
    """
    Extracts the provided 'features' column of the previous token for each token in the sentence

    Returns a list of dictionaries representing the features of the previous token, , else a placeholder '_' is added
    """
    features = []
    for i, token_info in enumerate(sentence):
        if i == 0:
            features.append('_')  #placeholder for the 1st token
        else:
            previous_token_features = sentence[i - 1]['features']
            features.append(previous_token_features)
    return features
