import numpy as np

def get_head_lemma(sentence, model):
    '''Function finds and returns a list of 200 dimension word embeddings of the head words for each token
    : param: sentence: list of dicts
    : param: model: word embedding model
    : return: list of word embedings
    '''
    list_of_head_lemmas = []
    for row in sentence:# iterate through words in sentence
        word_id = row['id'] # establish word id
        if '.' in word_id:# if punctuation return None
            head_lemma = None
        else:
            head_id = row['head']# establish id of the head
            if head_id == '0' or head_id == 0: # if id of head is 0 -> return the lemma of head
                head_lemma = row.get('lemma', None)
            else:
                # check for the line that has the head of the current line
                head_dict = next((item for item in sentence if item['id'] == head_id), None)
                if head_dict is not None:
                    # extract the lemma of the head
                    head_lemma = head_dict.get('lemma', None)
                else:
                    head_lemma = None
        list_of_head_lemmas.append(head_lemma)
        head_lemma_WE = []
        for item in list_of_head_lemmas: # check if the word is in the model
            if item in model and not None:
                head_lemma_WE.append(model[item]) # if word is in the model append the vector
            elif item not in model or None:
                head_lemma_WE.append(np.array([0.0] * 200)) # if word not in the model append 0*200
        if len(head_lemma_WE)!=len(sentence):
            print('ERROR')
        else:
            return head_lemma_WE  
