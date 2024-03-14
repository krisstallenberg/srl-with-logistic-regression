def extract_morph_new(sentence, dict_morph):

    features = []
    for token in sentence:


        dict_morph_copy=dict_morph.copy()
        for feature, value in token.items():
            if value !='':
                feature_key = f"{feature}={value}"
                if feature_key in dict_morph:
                    dict_morph_copy[feature_key] += 1
            else:
                continue
            morph_values=dict_morph_copy.values()
            features.append(morph_values)
    return features
