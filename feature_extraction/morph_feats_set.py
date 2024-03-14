def count_morphological_markers(data):
    """
    Counts all possible morphological markers in the given data.

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
    print(morphological_markers)
    return morphological_markers
