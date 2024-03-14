def find_all_morph_features(data):
    """
    Find all morphological feature key-value pairs. 

    Parameters:
    - data: A list of dicts, where each dict represents a sentence and contains
      tokens, each with a 'feats' dict of morphological features.

    Returns:
    - A dict with keys as unique morphological features (e.g., "Tense=present")
      and values all initialized to 0.
    """
    morphological_markers = {}
    marker_list=[]
    # Iterate through each sentence in the data
    for sentence in data:
        # Assuming each sentence contains a list of tokens
        for token_list in sentence:

            feats = token_list['features']

        # Iterate through the features in the feats dictionary
            for feature, value in feats.items():
            # Construct the feature representation as 'Feature=Value'
                feature_representation = f"{feature}={value}"
                marker_list.append(feature_representation)
            # Initialize the feature count to 0 if it's not already in the dict
                # if feature_representation not in morphological_markers:
                #     morphological_markers[feature_representation] = 0
    marker_set=set(marker_list)
    for element in marker_set:
        morphological_markers[element]=0
    return morphological_markers


def extract_current_morph(sentence, dict_morph):
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
