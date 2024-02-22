def am_i_punct(sentence):
    '''Function determines if the token is punctuation
    : param: sentence: list of dicts containing feature representation forneach token
    : return: am_i_punct_list: list of booleans
    '''
    
    am_i_punct_list=[]
    for item in sentence:
        if 'PUNCT' in item['upos']:
            am_i_punct_list.append(1)
        else:
            am_i_punct_list.append(0)
            
    if len(am_i_punct_list) != len(sentence):
        print('ERROR')
    else:
        return am_i_punct_list