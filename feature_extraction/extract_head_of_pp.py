def head_word_of_pp(sentence):
    """
    Extracts the head word of the first noun phrase inside each PP and appends the preposition information to the phrase type

    Returns a list containing features representing the head word of each PP
    """
    features = []  
    for token_info in sentence:
        if token_info['dependency_relation'] == 'case':  #it first checks if the token is a preposition
            pp_head_word = None
            for prep_head_relation in sentence:#looking for the head word of the pp
                if (prep_head_relation['head'] == token_info['id'] and #if the token's head matches the id of the pp token 
                        prep_head_relation['dependency_relation'] != 'punct'):  #and is not a punctuation mark
                    pp_head_word = prep_head_relation['form']#it stores it as the head word of the pp
                    break
            if pp_head_word:
                feature = f"PP-{token_info['form']}-{pp_head_word}"#it appends the preposition and the head to the feature
            else:
                feature = f"PP-{token_info['form']}" #otherwise it only includes the preposition
            features.append(feature)  
    return features


