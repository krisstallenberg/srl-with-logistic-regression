def extract_next_token_morph_features(sentence):
    """
    Extracts the provided 'features' column of the next token for each token in the sentence

    Returns a list of dictionaries representing the features of the next token, else a placeholder '_' is added
    """
    features = []
    for i, token_info in enumerate(sentence):
        if i == len(sentence) - 1:
            features.append('_')  #placeholder for the last token
        else:
            next_token_features = sentence[i + 1]['features']
            features.append(next_token_features)
    return features