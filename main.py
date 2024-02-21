# Import /feature_extraction/extract_embedding.py
import time
import sys
sys.path.append('feature_extraction')
import extract_embedding
import pandas as pd


def extract_features(dataframe):
    # dataframe['feature']['pos'] = extract_embedding(dataframe)
    pass

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

    # # Assuming 'path_train' is the path to your CoNLL-U file
    # with open(file_path, "r", encoding="utf-8") as file:
    #     sentences = file.read().strip().split('\n\n')

    # dfs = []  # This will store a DataFrame for each sentence

    # for sentence in sentences:
    #     lines = sentence.split('\n')
    #     sentence_data = []  # This will hold data for the current sentence
    #     for line in lines:
    #         if line.startswith('#'):
    #             continue  # Skip comment lines
    #         fields = line.split('\t')
    #         sentence_data.append(fields)
    #     # Convert sentence data to a DataFrame
    #     if sentence_data:
    #         df = pd.DataFrame(sentence_data)
    #         dfs.append(df)

    # dupl_sents = []
    # for df in dfs:
    #     preds=[]
    #     preds = [df.iloc[i][10] if '.' not in str(df.iloc[i][0]) else '_' for i in range(len(df))]
    #     count_of_pred=0
    #     for i, pred in enumerate(preds):
    #         if pred == '_':
    #             continue
    #         else:
    #             count_of_pred+=1
    #             arg_col_name = int(10 + count_of_pred)
    #             new_pred_list = ['_'] * len(preds)
    #             new_pred_list[i] = pred 
    #             df_v=pd.DataFrame()
    #             df_v=df.copy()
                
    #             if arg_col_name in df.columns:
    #                 columns_to_keep = [int(i) for i in range(0, 10)] + [arg_col_name]
    #                 df_v = df[columns_to_keep].copy()
    #                 df_v.insert(10, 10, new_pred_list)
    #                 df_v = df_v.set_axis(['ID','FORM','LEMMA','UPOS','XPOS','FEATURES','HEAD','DEPREL','DEPS','MISC','PRED','ARGS'], axis=1)
    #                 if len(df_v.columns)!=12:
    #                     print(df_v)
    #             dupl_sents.append(df_v)

    # indices_to_remove = [i for i, df in enumerate(dupl_sents) if 'ID' not in df.columns]
    # for i in sorted(indices_to_remove, reverse=True):
    #     dupl_sents.pop(i)
        
    # return dupl_sents

    sentences = []
    with open(file_path, 'r') as file:
        current_tokens = []
        sent_id = None
        text = None
        for line in file:
            if line.startswith('#'):
                continue
            else:
                if line.strip() == '':
                    sentences.append(sentence)
                    sentence = []
                else:
                    line = line.strip().split('\t')
                    # Split the features into a dictionary
                    features = dict()
                    for feature in line[5].split('|'):
                        key_value_pair = feature.split('=')
                    # Check if the split result is valid (contains both key and value)
                        if len(key_value_pair) == 2:
                            key, value = key_value_pair
                            features[key] = value 
                    sentence.append({
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

    # Extract features from the data
    start_time = print_process("extracting features")
    extract_features(dev_data)
    extract_features(train_data)
    extract_features(test_data)
    print_process("extracting features", start_time)

    # Train the model
    start_time = print_process("training")
    model = train_model(train_data)
    print_process("training", start_time)

    print(dev_data[-1])

    # Infer the model
    start_time = print_process("inference")
    results = infer_model(model, dev_data)
    print_process("inference", start_time)

# Run main function when called from CLI
if __name__ == "__main__":
    main()