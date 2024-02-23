from nltk.corpus import propbank

def extract_propbank_args(sentences):
    """
    Extract features for each sentence based on PropBank arguments.
    Return a list of features for each sentence, where each feature is a list of strings representing possible PropBank arguments for the predicate in the sentence.

    sentences (list): a Python dictionary containing all the columns from CoNLLu file for each token.
    """

    roles_dict = {} #creating an empty dict where the roles per predicate will be stored
    features = [] #empty list where the features will be stored
    for dictionary in sentences: #iterating over the dictionary of info per token in the sentence
        i = len(dictionary) #setting the length of the sentence
        predicate = dictionary['predicate'] #getting the predicate of the sentence
        if predicate != '_': #skipping tokens with predicate '_'
            try: #checking if the predicate of the sentence exists in the roleset, if it does, the roles are searched for
                pred = propbank.roleset(predicate) ### as seen on NLTK documentation https://www.nltk.org/howto/propbank.html#propbank on Feb 22
                roles = pred.findall("roles/role")
                for role in roles:
                    if predicate not in roles_dict: #if the key doesn't exist, adding the roles to the roles dictionary
                        roles_dict[predicate] = set(role.attrib['n']) 
                    else: #if it does, adding the as a set to avoid duplicates
                        roles_dict[predicate].add(role.attrib['n']) 
            except ValueError: #if the predicate is not in propbank frames, it throws an error
                roles_dict[predicate] = set('x') #add a 'x' to the dictionary
        
            if predicate in roles_dict: #if the predicate is in the dictionary
                sorted_list = sorted(roles_dict[predicate]) #ordering the items in the set value
                arg_feature = '_'.join(sorted_list) #and joining them in a string to represent the arguments possible
            
                features.extend([arg_feature] *  i) #the feature will be the size of the sentence
            else:
                features.extend((['x'] *  i)) #if the predicate is not in dict, it will be just x
    
    return features