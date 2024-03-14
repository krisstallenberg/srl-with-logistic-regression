def extract_morph_new(sentence, dict_morph):
    """
    Creates a one-hot encoding for all possible key-value pairs of morphological features.

    Returns a list of list of integers.

    sentence (list of dicts): list of dictionaries, one for each token in the sentence.
    dict_morph: dictionary of feature key-value combinations.
    """

    features = []

    # Iterate over all tokens .
    for token in sentence:

        # Initialize empty list for this token's features.
        token_features = []

        # Iterate over all feature key-value pairs.
        for feature_combination in dict_morph.keys():

            # Iterate over all token's feature key-value pairs.
            for feature, value in token['features'].items():
                token_feature_combination = f"{feature}={value}"

                # If the key-value pairs match, add 1 to the token_features list, otherwise add 0.
                if token_feature_combination == feature_combination:
                    token_features.append(1)
                else:
                    token_features.append(0)

        # Append the list of this token's features to the list of lists.
        feature.append(token_features)
    return features
