import pickle
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer


def load_data(file_path):
    with open(file_path.replace('.conllu', '.preprocessed.pkl'), 'rb') as pickle_file:
        return pickle.load(pickle_file)


def prepare_for_model(data):
    """Convert feature dict to pandas DataFrame to handle data for training.
    Return a pandas df with the relevant features for training the model.

    Paramenters:
    -data: a list of objects, where each object represents one 'frame' in a sentence.
    """
    list_features = [] #creating an empty list where the data will be stored
    
    for sentences in data:
        for token_dict in sentences:
            dict_feat = token_dict['features'] #grabbing the nested dictionaries where the features are stored
            list_features.append(dict_feat) #appending the dict to the list

    df = pd.DataFrame(list_features) #converting the list of dictionaries into a pandas dataframe, as seen at https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe

    #selecting the features that are needed for the model, which are now columns in the df
    df = df[['embedding', 
         'pos_extracted', 
         'position_rel2pred',
         'embedding_head',
         'num_of_children',
         'punct_extracted',
         'head_pos',
         'dep_path',
         'cosine_similarity_w_predicate',
        'pos_misc_feature',
        'head_pp_feature',
        'ner',
        'propbank_arg']] 

    return df


def extract_gold_labels(data_file):
    '''
    Extract gold labels.
    Return a list of gold labels.
    
    :param data_file: a list of objects, where each object represents one 'frame' in a sentence.
    :type data_file: string
    '''
    labels = []
    
    for sentence in data_file:
        for token_dict in sentence:
            #adding gold label to labels
            gold_label = token_dict['argument']
            labels.append(gold_label)

    return labels


def df_to_dict(dataframe):
    """Convert a pandas dataframe to a python dictionary.
    Return a dictionary containing the name of the feature as key and the respective feature as value.

    Parameter:
    -dataframe: dataframe containing the features for the model.
    """
    data_dict = dataframe.to_dict(orient='records')
    return data_dict


def extract_feature_values(row_dict, selected_features):
    '''
    Extract feature value pairs from row
    
    :param row: row from conllu file
    :param selected_features: list of selected features
    :type row: string
    :type selected_features: list of strings
    
    :returns: dictionary of feature value pairs
    '''
    feature_to_index = {'embedding': 0,
                        'pos_extracted': 1, 
                        'position_rel2pred': 2, 
                        'embedding_head': 3,
                        'num_of_children': 4, 
                        'punct_extracted': 5, 
                        'head_pos': 6, 
                        'dep_path': 7,
                        'cosine_similarity_w_predicate': 8, 
                        'pos_misc_feature': 9,
                        'head_pp_feature': 10,
                        'ner': 11, 
                        'propbank_arg': 12}
    
    feature_values_dict = {}
    for feature_name in selected_features:
        r_index = feature_to_index.get(feature_name)
        feature_values_dict[feature_name] = row_dict.get(feature_name)

    return feature_values_dict

def create_vectorizer_traditional_features(feature_values):
    '''
    Create vectorizer for set of feature values
    
    :param feature_values: list of dictionaries containing feature-value pairs
    :type feature_values: list of dictionairies (key and values are strings)
    
    :returns: vectorizer with feature values fitted
    '''
    vectorizer = DictVectorizer()
    vectorizer.fit(feature_values)
    
    return vectorizer
        
    
def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''
    Take sparse and dense feature representations and appends their vector representation
    
    :param dense_vectors: list of dense vector representations
    :param sparse_features: list of sparse vector representations
    :type dense_vector: list of arrays
    :type sparse_features: list of lists
    
    :returns: list of arrays in which sparse and dense vectors are concatenated
    '''
    
    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())

    
    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,dense_vectors[index]))  
        combined_vectors.append(combined_vector)
    
    
    return combined_vectors
    

def extract_features(data_file, vectorizer=None):
    '''
    Extract features and converts them into a vector.
    Return a list of vector representation of tokens.
    
    :param data_file: a list of objects, where each object represents one 'frame' in a sentence.
    :type data_file: string
    
    '''

    dense_vectors = []
    traditional_features = []

    selected_features = [ 
                     'pos_extracted',
                     'position_rel2pred',
                     'head_pos',
                     'dep_path',
                     'pos_misc_feature',
                     'ner',
                    ] 
    
    df = prepare_for_model(data_file) #extracting the features necessary from the data file by converting to a pandas df
    features_dict_list = df_to_dict(df) #converting the df back to dictionaries to extract the features and convert to vector representation
    
    for token_dict in features_dict_list:
        lemma_vector = token_dict['embedding']
        head_vector = token_dict['embedding_head']
        #cos_sim_vector = np.asarray([token_dict['cosine_similarity_w_predicate']]) #converting the numerical feat to np arrays to be concatenated in a single array
        #num_children = np.asarray([token_dict['num_of_children']])
        punct_extracted = np.asarray([token_dict['punct_extracted']])
        dense_vectors.append(np.concatenate((lemma_vector,head_vector,punct_extracted))) #contactenating embeddings plus numerical value features
        #mixing very sparse representations (for one-hot tokens) and dense representations is a bad idea
        #we thus only use other features with limited values
        other_features = extract_feature_values(token_dict, selected_features)
        traditional_features.append(other_features)

    
    #create vector representation of traditional features
    if vectorizer is None:
        #creates vectorizer that provides mapping (only if not created earlier)
        vectorizer = create_vectorizer_traditional_features(traditional_features)
    sparse_features = vectorizer.transform(traditional_features)
    combined_vectors = combine_sparse_and_dense_features(dense_vectors, sparse_features)
    
    return combined_vectors, vectorizer


def create_classifier(features, labels):
    '''
    Create classifier from features represented as vectors and gold labels
    
    :param features: list of vector representations of tokens
    :param labels: list of gold labels
    :type features: list of vectors
    :type labels: list of strings
    
    :returns trained logistic regression classifier
    '''
    
    
    lr_classifier = LogisticRegression(solver='saga')
    lr_classifier.fit(features, labels)
    
    return lr_classifier


def label_data(vec,testfile, classifier):
    '''
    Extract features and gold labels from test data and runs a classifier
    
    :param testfile: a list of objects, where each object represents one 'frame' in a sentence.
    :param classifier: trained classifier
    :type testfile: string
    :type classifier: LogisticRegression
    
    :return predictions: list of predicted labels
    :return labels: list of gold labels
    '''
    
    dense_feature_representations,vect = extract_features(testfile,vec)
    labels = extract_gold_labels(testfile)
    predictions = classifier.predict(dense_feature_representations)
    
    return labels,predictions



def train_model(train_data):
	
    dense_feature_representations,vec = extract_features(train_data)
    gold = extract_gold_labels(train_data)
    model = create_classifier(dense_feature_representations, gold)
    
    return model, vec

def test_model(test_data, model, vec):

    gold_labels,predicted = label_data(vec, test_data, model)
    
    return predicted, gold_labels
