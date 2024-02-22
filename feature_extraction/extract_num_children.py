def how_many_children(sentence):
    ''' Function finds how many subordinate dependecy relations does a token have.
    : param: sentence: list of dicts containing feature representation forneach token
    : return: list of integers
    '''
    list_of_children_num=[]
    for item in sentence:
        token_id=item['id']  # establish the id of the token
        count_of_children=0  # create count variable
        for word in sentence:
            if word['head']==token_id:   # if item is the head of word -> add 1 to count
                count_of_children+=1
        list_of_children_num.append(count_of_children)
    if len(list_of_children_num)!=len(sentence): # checking for matching dimensions
        print('ERROR')
    else:
        return list_of_children_num