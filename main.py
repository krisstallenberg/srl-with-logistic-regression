# Import /feature_extraction/extract_embedding.py
import time
import sys
sys.path.append('feature_extraction')
import extract_embedding

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
    with open(file_path, 'r') as file:
        current_tokens = []
        sent_id = None
        text = None
        for line in file:
            if line.startswith('#'):
                if line.startswith('# sent_id'):
                    sent_id = line.split('=')[1].strip()
                elif line.startswith('# text'):
                    text = line.split('=')[1].strip()
            else:
                if line.strip() == '':
                    sentence = {
                        'sent_id': sent_id,
                        'text': text,
                        'tokens': current_tokens
                    }
                    sentences.append(sentence)
                    current_tokens = []
                else:
                    line = line.strip().split('\t')
                    # Features is a string of key-value pairs separated by '|' (the keys and values are separated by '=') Make sure to split this into a dictionary.
                    features = dict()
                    for feature in line[5].split('|'):
                        key_value_pair = feature.split('=')
                    # Check if the split result is valid (contains both key and value)
                        if len(key_value_pair) == 2:
                            key, value = key_value_pair
                            features[key] = value 
                    current_tokens.append({
                        'id': line[0],
                        'form': line[1],
                        'lemma': line[2],
                        'upos': line[3],
                        'xpos': line[4],
                        'features': features,
                        'head': line[6],
                        'dependency_relation': line[7],
                        'dependency_graph': line[8],
                        'miscellaneous': line[9]
                    })
    return sentences

def print_process(process_name, start_time=None):
    if start_time is None:
        print(f"{process_name.capitalize()}...")
        return time.time()
    else:
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 1)
        print(f"Successfully finished {process_name.lower()} in {elapsed_time} seconds!")

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