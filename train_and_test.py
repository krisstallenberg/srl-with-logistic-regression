import pickle
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer


def load_data(file_path):
	with open(file_path.replace('.conllu', '.preprocessed.pkl'), 'rb') as pickle_file:
		return pickle.load(pickle_file)
		
def get_all_sentences(data):
	word_list = []
	for sents in data:
		for word in sents:
			for f in word['features']:
				word[f]=word['features'][f]
			word_list.append(word)
	return word_list
	
def df_to_dict(dataframe):
	data_dict = dataframe.to_dict(orient='records')
	for row in data_dict:
		row['embedding'] = str(row['embedding'])
		row['cosine_similarity_w_predicate'] = str(row['cosine_similarity_w_predicate'])
	return data_dict
		

def train_model(train_data):
	
	train_df = pd.DataFrame(get_all_sentences(train_data))
	
	train_df = train_df.drop(columns=['id','features', 'Definite', 'PronType', 'Number', 'Mood','Person', 'Tense', 'VerbForm','head','dependency_graph',
								  'miscellaneous','head_pp_feature','prev_token_morph_features', 'next_token_morph_features','embedding_head', 'punct_extracted',
								  'NumType','Degree', 'Case', 'Gender', 'Poss', 'Voice', 'Foreign', 'Reflex', 'Typo','num_of_children','Abbr','propbank_arg'])

	train_golds = train_df['argument'].copy()
	train_df= train_df.drop(columns =['argument'])
	
	train_data_dict  = df_to_dict (train_df)
	vec = DictVectorizer()
	logreg = LogisticRegression()
	X_transformed = vec.fit_transform(train_data_dict)
	model = logreg.fit(X_transformed, train_golds)
	
	return model, vec

def test_model(test_data, model, vec):
	
	test_df = pd.DataFrame(get_all_sentences(test_data))
	
	test_df = test_df.drop(columns=['id','features', 'Definite', 'PronType', 'Number', 'Mood','Person', 'Tense', 'VerbForm','head','dependency_graph',
								  'miscellaneous','head_pp_feature','prev_token_morph_features', 'next_token_morph_features','embedding_head', 'punct_extracted',
								  'NumType','Degree', 'Case', 'Gender', 'Poss', 'Voice', 'Foreign', 'Reflex', 'Typo','num_of_children','Abbr','propbank_arg'])

	test_golds = test_df['argument'].copy()
	test_df= test_df.drop(columns =['argument'])
	
	test_data_dict  = df_to_dict (test_df)
	X_transformed = vec.transform(test_data_dict)
	predicts = model.predict(X_transformed)
	
	return predicts, test_golds
