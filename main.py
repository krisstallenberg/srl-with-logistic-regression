import time
import json
import argparse
import sys
import pandas as pd
import gensim.downloader as api
sys.path.append('feature_extraction')
from extract_we_lemmas import extract_embedding_lemma
from extract_pos_token import extract_pos_token
from extract_position_rel2pred import extract_word_position_related_to_predicate
from extract_lemma_head_WE import get_head_lemma
from extract_num_children import how_many_children
from extract_punct_data import am_i_punct
from extract_pos_head import extract_UPOS_of_head
from extract_dependency_path import extract_dependency_path
from extract_cosine_similarity_w_predicate import extract_cosine_similarity_w_predicate
from extract_pos_with_misc_spacing import pos_misc_feature
from extract_head_of_pp import head_word_of_pp 
from extract_ner import extract_ner
from extract_propbank_possible_roles import extract_propbank_args
from extract_current_morph import extract_current_morph, find_all_morph_features
from extract_predicate import extract_predicate
from evaluation import evaluation_model
from train_and_test import train_model, test_model

def extract_features(data,model):

	morph_features_dict = find_all_morph_features(data)

	for sentence in data:

		features_from_dataset = extract_current_morph(sentence, morph_features_dict)
		for token, morph_features_list in zip(sentence, features_from_dataset):
			token['features']['morph_features'] = morph_features_list

		embeddings = extract_embedding_lemma(sentence,model)
		for token, embedding in zip(sentence, embeddings):
			token['features']['embedding'] = embedding

		pos_tokens = extract_pos_token(sentence)
		for token, pos in zip(sentence, pos_tokens):
			token['features']['pos_extracted'] = pos

		positions_rel2pred = extract_word_position_related_to_predicate(sentence)
		for token, pos_rel2pred in zip(sentence, positions_rel2pred):
			token['features']['position_rel2pred'] = pos_rel2pred

		WE_head_emb = get_head_lemma(sentence, model)
		for token, WE_head_embedding in zip(sentence, WE_head_emb):
			token['features']['embedding_head'] = WE_head_embedding

		num_of_children = how_many_children(sentence)
		for token, num_of_child in zip(sentence, num_of_children):
			token['features']['num_of_children'] = num_of_child

		list_punct = am_i_punct(sentence)
		for token, puncts in zip(sentence, list_punct):
			token['features']['punct_extracted'] = puncts
		
		h_pos = extract_UPOS_of_head(sentence)
		for token, head_pos in zip(sentence, h_pos):
			token['features']['head_pos'] = head_pos

		d_paths = extract_dependency_path(sentence)
		for token, dep_path in zip(sentence, d_paths):
			token['features']['dep_path'] = dep_path

		cosine_similarity_w_predicate = extract_cosine_similarity_w_predicate(sentence, model)
		for token, cosine_sim in zip(sentence, cosine_similarity_w_predicate):
			token['features']['cosine_similarity_w_predicate'] = cosine_sim

		pos_misc_features = pos_misc_feature(sentence)
		for token, pos_misc in zip(sentence, pos_misc_features):
			token['features']['pos_misc_feature'] = pos_misc

		head_pp_features = head_word_of_pp(sentence)
		for token, head_pp_feature in zip(sentence, head_pp_features):
			token['features']['head_pp_feature'] = head_pp_feature
	
		ner_tags = extract_ner(sentence)
		for token, ner_tag in zip(sentence, ner_tags):
			token['features']['ner'] = ner_tag

		propbank_args = extract_propbank_args(sentence)
		for token, propbank_arg in zip(sentence, propbank_args):
			token['features']['propbank_arg'] = propbank_arg

		predicates = extract_predicate(sentence)
		for token,predicate in zip(sentence, predicates):
			token['features']['predicate'] = predicate

	return data

def preprocess_data(file_path):
	"""
	Parses the CoNNL-U Plus file and returns a list of sentences.
	Extract features from the data and return a list of objects.

	Returns a list of objects, where each object represents one 'frame' in a sentence.

	data (str): The file path to the data to be preprocessed.
	"""

	sentences = []
	sentence = []  # Initialize an empty list for the current sentence
	with open(file_path, 'r', encoding="utf8") as file:
		for line in file:
			line = line.strip().split('\t')
			if line[0].startswith('#'):
				# If the line starts with '#', it's a comment, ignore it
				continue
			elif line[0].strip() != '':
				# Split the features string into a dictionary
				features = dict()
				for feature in line[5].split('|'):
					key_value_pair = feature.split('=')

					# Check if the split result is valid, if it is, add it to the dictionary
					if len(key_value_pair) == 2:
						key, value = key_value_pair
						features[key] = value

				# Create a token if its ID does not contain a period
				if '.' not in line[0] and len(line) > 10:
					token = {
						'id': line[0],
						'form': line[1],
						'lemma': line[2],
						'upos': line[3],
						'xpos': line[4],
						'features': features,
						'head': line[6],
						'dependency_relation': line[7],
						'dependency_graph': line[8],
						'miscellaneous': line[9],
						'predicate': line[10],
						'argument': line[11:]  # Store all remaining elements as arguments.
					}
					# Append the token to the sentence.
					sentence.append(token)

			# A new line indicates the end of a sentence.
			elif line[0].strip() == '':
				# Append the completed sentence to the sentences list.
				sentences.append(sentence)
				# Reset sentence for the next sentence.
				sentence = []

	# Iterate over all sentences. Create copies of sentences for each predicate.
	expanded_sentences = []
	for sentence in sentences:
		# Find all predicates in the sentence.
		predicates = [token['predicate'] for token in sentence if token['predicate'] != '_']

		# for every predicate, create a copy of the sentence.
		for index, predicate in enumerate(predicates):
			sentence_copy = [token.copy() for token in sentence]
			for token in sentence_copy:
				# Keep only this predicate.
				if token['predicate'] != predicate:
					token['predicate'] = '_'

				# Keep only the relevant argument for this predicate. Overwrite 'V' and 'C-V' with '_'.
				token['argument'] = '_' if token['argument'][index] in ['V', 'C-V'] else token['argument'][index]

			# Append only sentences with arguments.
			if any(token['argument'] != '_' for token in sentence_copy):	
				expanded_sentences.append(sentence_copy)

	return expanded_sentences




def print_process(process_name, start_time=None):
	"""
	Print the process name and the elapsed time.

	process_name (str): The name of the process
	start_time (float): The start time of the process, for second call
	"""


	if start_time is None:
		print(f"{process_name.capitalize()}...")
		return time.time()
	else:
		end_time = time.time()
		elapsed_time = round(end_time - start_time, 1)
		print(f"Successfully finished {process_name.lower()} in {elapsed_time} seconds!\n")

import pickle

def save_data(data, file_path):
	with open(file_path.replace('.conllu', '.preprocessed.pkl'), 'wb') as pickle_file:
		pickle.dump(data, pickle_file)

def load_data(file_path):
	with open(file_path.replace('.conllu', '.preprocessed.pkl'), 'rb') as pickle_file:
		return pickle.load(pickle_file)

def main():
	"""
	Main function for Semantic Role Labeling (SRL).
	Uses the Universal PropBank dataset.

	1. Preprocess the data and extract features
	2. Train a model
	3. Evaluate the model
	"""
	# Parse command line arguments.
	parser = argparse.ArgumentParser(description='Semantic Role Labeling (SRL) using the Universal PropBank dataset.')
	parser.add_argument('--use-cached', action='store_true' , help='Use cached preprocessed files for training and inference.')
	args = parser.parse_args()

	# Define file paths.
	dev_file_path = 'data/en_ewt-up-dev.conllu'
	train_file_path = 'data/en_ewt-up-train.conllu'
	test_file_path = 'data/en_ewt-up-test.conllu'

	# By default, the program preprocesses the data, extracts features and stores the data.
	if not args.use_cached:
		print("Not using cached files. Preprocessing, extracting features and storing data...\n")

		# Preprocess the data.
		start_time = print_process("preprocessing")
		dev_data = preprocess_data(dev_file_path)
		train_data = preprocess_data(train_file_path)
		test_data = preprocess_data(test_file_path)
		print_process("preprocessing", start_time)

		# Download the model and assign to variable.
		start_time = print_process("loading language model")
		model = api.load("glove-wiki-gigaword-200")   ### as seen at https://github.com/piskvorky/gensim-data on 21st Feb 2024
		print_process("loading language model", start_time)

		# Extract features from the data.
		start_time = print_process("extracting features")
		extract_features(dev_data,model)
		extract_features(train_data,model)
		extract_features(test_data,model)
		print_process("extracting features", start_time)

		# Store all preprocesed data with extracted features as JSON.
		start_time = print_process("storing data with extracted features as Pickle")
		save_data(dev_data, dev_file_path)
		save_data(train_data, train_file_path)
		save_data(test_data, test_file_path)
		print_process("storing data with extracted features as Pickle", start_time)

	# If --use-cached is set, load the preprocessed data with extracted features with Pickle.
	else:
		start_time = print_process("loading data with extracted features with Pickle")
		dev_data = load_data(dev_file_path)
		train_data = load_data(train_file_path)
		test_data = load_data(test_file_path)
		print_process("loading data with extracted features with Pickle", start_time)

	# Print first token of test data
	for sentence in test_data:
		for word in sentence:
			for key, value in word.items():
				print(f"key: {key}: value: {value}")
			break

	# Train the model
	start_time = print_process("training")
	model, vec= train_model(train_data)
	print_process("training",start_time)

	# Predict with model
	start_time = print_process("inferencing")
	results, golds = test_model(test_data,model,vec)
	print_process("inferencing", start_time)

    # evaluation the model
	start_time = print_process("evaluating")
	evaluation_model(golds, results)
	print_process("evaluating", start_time)

# Run main function when called from CLI
if __name__ == "__main__":
	main()
