import pickle
import pprint  # Add this line
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def load_data(file_path):
    with open(file_path.replace('.conllu', '.preprocessed.pkl'), 'rb') as pickle_file:
        return pickle.load(pickle_file)

dev_file_path = 'data/en_ewt-up-dev.conllu'
train_file_path = 'data/en_ewt-up-train.conllu'
test_file_path = 'data/en_ewt-up-test.conllu'

dev_data = load_data(dev_file_path)
train_data = load_data(train_file_path)
test_data = load_data(test_file_path)

def extract_feature_values(row_dict, selected_features):
    '''
    Function that extracts feature value pairs from row
    
    :param row: row from conllu file
    :param selected_features: list of selected features
    :type row: string
    :type selected_features: list of strings
    
    :returns: dictionary of feature value pairs
    '''
    feature_to_index = {'embedding': 0, 
                        'pos': 1, 
                        'position_rel2pred': 2, 
                        'embedding_head': 3, 
                        'num_of_children': 4, 
                        'punct': 5,
                        'head_pos': 6,
                        'dep_path': 7,
                        'cosine_similarity_w_predicate': 8,
                        'pos_misc_feature': 9,
                        'ner': 10,
                        'propbank_arg': 11}
    
    feature_values_dict = {}
    for feature_name in selected_features:
        r_index = feature_to_index.get(feature_name)
        feature_values_dict[feature_name] = row_dict.get(feature_name)

    return feature_values_dict
    
    
def create_vectorizer_traditional_features(feature_values):
    '''
    Function that creates vectorizer for set of feature values
    
    :param feature_values: list of dictionaries containing feature-value pairs
    :type feature_values: list of dictionairies (key and values are strings)
    
    :returns: vectorizer with feature values fitted
    '''
    vectorizer = DictVectorizer()
    vectorizer.fit(feature_values)
    
    return vectorizer
        
    
def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''
    Function that takes sparse and dense feature representations and appends their vector representation
    
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
    

def extract_features_plus_gold_labels(data_file, vectorizer=None):
    '''
    Function that extracts traditional features as well as embeddings and gold labels using word embeddings for current and preceding token
    
    :param conllfile: path to conll file
    :type conllfile: string
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    labels = []
    dense_vectors = []
    traditional_features = []

    selected_features = [ 
                     'pos',
                     'position_rel2pred',
                     'punct',
                     'head_pos',
                     'dep_path',
                     'pos_misc_feature',
                     'ner',
                     'propbank_arg'
                    ] 

    for sentence in data_file:
        for token_dict in sentence:
            features_dict = token_dict['features']
            lemma_vector = features_dict['embedding']
            head_vector = features_dict['embedding_head']
            cos_sim_vector = features_dict['cosine_similarity_w_predicate']
            num_children = features_dict['num_of_children']
            dense_vectors.append(np.concatenate((lemma_vector,head_vector,cos_sim_vector,num_children)))
            #mixing very sparse representations (for one-hot tokens) and dense representations is a bad idea
            #we thus only use other features with limited values
            other_features = extract_feature_values(features_dict, selected_features)
            traditional_features.append(other_features)
            #adding gold label to labels
            gold_label = token_dict['argument']
            labels.append(gold_label)

    
    #create vector representation of traditional features
    if vectorizer is None:
        #creates vectorizer that provides mapping (only if not created earlier)
        vectorizer = create_vectorizer_traditional_features(traditional_features)
    sparse_features = vectorizer.transform(traditional_features)
    combined_vectors = combine_sparse_and_dense_features(dense_vectors, sparse_features)
    
    return combined_vectors, vectorizer


def create_classifier(features, labels):
    '''
    Function that creates classifier from features represented as vectors and gold labels
    
    :param features: list of vector representations of tokens
    :param labels: list of gold labels
    :type features: list of vectors
    :type labels: list of strings
    
    :returns trained logistic regression classifier
    '''
    
    
    lr_classifier = LogisticRegression(solver='saga')
    lr_classifier.fit(features, labels)
    
    return lr_classifier

def label_data(testfile, classifier):
    '''
    Function that extracts word embeddings as features and gold labels from test data and runs a classifier
    
    :param testfile: path to test file
    :param classifier: trained classifier
    :type testfile: string
    :type classifier: LogisticRegression
    
    :return predictions: list of predicted labels
    :return labels: list of gold labels
    '''
    
    dense_feature_representations, labels = extract_features_plus_gold_labels(testfile)
    predictions = classifier.predict(dense_feature_representations)
    
    return predictions, labels

print('Extracting dense features...')
dense_feature_representations, labels = extract_features_plus_gold_labels(train_data)
print('Training classifier....')
classifier = create_classifier(dense_feature_representations, labels)
print('Running evaluation...')
predicted, gold = label_data(dev_data, classifier)
print_confusion_matrix(predictions, goldlabels)
print_precision_recall_fscore(predicted, gold)