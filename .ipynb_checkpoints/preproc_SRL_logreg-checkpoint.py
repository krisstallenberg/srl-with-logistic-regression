import pandas as pd

def preprocess_data(file_path):

    # Assuming 'path_train' is the path to your CoNLL-U file
    with open(file_path, "r", encoding="utf-8") as file:
        sentences = file.read().strip().split('\n\n')

    dfs = []  # This will store a DataFrame for each sentence

    for sentence in sentences:
        lines = sentence.split('\n')
        sentence_data = []  # This will hold data for the current sentence
        for line in lines:
            if line.startswith('#'):
                continue  # Skip comment lines
            fields = line.split('\t')
            sentence_data.append(fields)
        # Convert sentence data to a DataFrame
        if sentence_data:
            df = pd.DataFrame(sentence_data)
            dfs.append(df)

    # Concatenate all sentence DataFrames
    print(len(dfs))

    dupl_sents = []
    for df in dfs:
        preds=[]
        preds = [df.iloc[i][10] if '.' not in str(df.iloc[i][0]) else '_' for i in range(len(df))]
        count_of_pred=0
        for i, pred in enumerate(preds):
            if pred == '_':
                continue
            else:
                count_of_pred+=1
                arg_col_name = int(10 + count_of_pred)
                new_pred_list = ['_'] * len(preds)
                new_pred_list[i] = pred 
                df_v=pd.DataFrame()
                df_v=df.copy()
                
                if arg_col_name in df.columns:
                    columns_to_keep = [int(i) for i in range(0, 10)] +[arg_col_name]
                    df_v = df[columns_to_keep].copy()
                    df_v.insert(10, 10, new_pred_list)
                    df_v = df_v.set_axis(['ID','FORM','LEMMA','UPOS','XPOS','FEATURES','HEAD','DEPREL','DEPS','MISC','PRED','ARGS'], axis=1)
                    if len(df_v.columns)!=12:
                        print(df_v)
                dupl_sents.append(df_v)

    indices_to_remove = [i for i, df in enumerate(dupl_sents) if 'ID' not in df.columns]
    for i in sorted(indices_to_remove, reverse=True):
        dupl_sents.pop(i)
        
    return dupl_sents