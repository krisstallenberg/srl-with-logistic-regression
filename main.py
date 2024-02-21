# Import /feature_extraction/extract_embedding.py
import time
import sys
sys.path.append('feature_extraction')
from extract_embedding import extract_embedding
import pandas as pd


def extract_features(data):
    for sentence in data:
        
        # For every extraction function, add these three lines, adjust the feature name and the function call accordingly ;)
        embeddings = extract_embedding(sentence)
        for token, embedding in zip(sentence, embeddings):
            token['features']['embedding'] = embedding

    return data

def train_model(data):
    pass

def infer_model(model, data):
    pass

def preprocess_data(file_path):
    """
    Extract features from the data and return a list of objects.

    Returns a list of objects, where each object represents a sentence in the data.

    data (str): The data to be preprocessed
    """

    sentences = []
    sentence = []  # Initialize an empty list for the current sentence
    with open(file_path, 'r') as file:
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
                if '.' not in line[0]:
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
                        'argument': line[10:]  # Store all remaining elements as arguments
                    }
                    # Append the token to the sentence
                    sentence.append(token)

            elif line[0].strip() == '':
                # Append the completed sentence to the sentences list
                sentences.append(sentence)
                # Reset sentence for the next sentence
                sentence = []

    # Iterate over all sentences. Create copies of sentences for each argument
    expanded_sentences = []
    for sentence in sentences:
        # Get the number of arguments for the first token
        num_arguments = len(sentence[0]['argument'])
        for arg_index in range(num_arguments):
            sentence_copy = [token.copy() for token in sentence]
            for token in sentence_copy:
                token['argument'] = token['argument'][arg_index]
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

def main():
    """
    Main function for Semantic Role Labeling (SRL).
    Uses the Universal PropBank dataset.

    1. Preprocess the data and extract features
    2. Train a model
    3. Evaluate the model
    """
    
    # Define file paths
    dev_file_path = 'data/en_ewt-up-dev.conllu'
    train_file_path = 'data/en_ewt-up-train.conllu'
    test_file_path = 'data/en_ewt-up-test.conllu'
    
    # Preprocess the data
    start_time = print_process("preprocessing")
    dev_data = preprocess_data(dev_file_path)
    train_data = preprocess_data(train_file_path)
    test_data = preprocess_data(test_file_path)
    print_process("preprocessing", start_time)

    # Extract features from the data
    start_time = print_process("extracting features")
    extract_features(dev_data)
    extract_features(train_data)
    extract_features(test_data)
    print_process("extracting features", start_time)

    # This is just to show the structure of the data. Delete this later.
    for sentence in dev_data[:2]:
        for word in sentence:
            for key, value in word.items():
                print(f"{key}: {value}")
            print("\n")

    # Train the model
    start_time = print_process("training")
    model = train_model(train_data)
    print_process("training", start_time)

    # Infer the model
    start_time = print_process("inference")
    results = infer_model(model, dev_data)
    print_process("inference", start_time)

# Run main function when called from CLI
if __name__ == "__main__":
    main()