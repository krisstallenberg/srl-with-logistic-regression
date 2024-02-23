# I used 'provide_confusion_matrix' , 'calculate_precision_recall_f1score' , 'evaluation_model' functions from previous term
# But I add 'extract_golds_from_data' to pull out Gold labels from input data for this task

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

# extract gold list from data
def extract_golds_from_data(data):
    """
    extract gold label from input data.

    Parameters
    ----------
    data 
        Train or test or dev dataset (after extracting fetures)
    
    Returns
    -------
        List of Gold labels
    """
    labels_train = []
    # for each sentence
    for sentence in data:
        # for each word in sentence
        for word in sentence:
            # for each feature of word
            for item in word.items():
                # if feature is argument 
                if item[0] == 'argument':
                    # append feature value to list of labels
                    labels_train.append(item[1])
    return labels_train

def provide_confusion_matrix(GoldLabel, PredictLabel, label_set):
    """
    use `sklearn.metric confusion_matrix` to create confusion matrix of model predict.
    and `sklearn.metric ConfusionMatrixDisplay` to display created confusion matrix.

    Parameters
    ----------
    GoldLabel : list
        list of all Gold labels
    PredictLabel : list
        list of all Prediction labels
    label_set : list 
        list of all classes
    
    Returns
    -------
        Confusion matrix
    """
    cf_matrix = confusion_matrix(GoldLabel, PredictLabel) # create a confusion matrix with gold and predicts
    print(cf_matrix) # print confusion_matrix as text
    display = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=label_set) # create graphical confusion_matrix
    fig, ax = plt.subplots(figsize=(15,15)) # create bigger plot because there is many classes in this task
    display.plot(ax =ax) # show confusion_matrix
    plt.xticks(rotation=90) # rotate X label of plot 90 degree
    plt.show() # show confusion matrix
    return cf_matrix # return confusion_matrix (maybe useful later)

def calculate_precision_recall_f1score(GoldLabel, PredictLabel, label_set): # function get gold and predict and set of labels
    """
    use `sklearn.metric classification_report` to get report of model predict.
    
    Parameters
    ----------
    GoldLabel : list
        list of all Gold labels
    PredictLabel : list
        list of all Prediction labels
    label_set : list 
        list of all classes
    
    Returns
    -------
        Classification report
    """
    report = classification_report(GoldLabel, PredictLabel, digits = 3, target_names=label_set) # calculate report
    print(report) # print report
    return report # return report (maybe useful later)

# def evaluation_model(GoldLabel, PredictLabel): # get gold and predict
def evaluation_model(data, PredictLabel): # get gold and predict
    """
    Evaluation models by call `calculate_precision_recall_f1score` and `provide_confusion_matrix` functions.

    Parameters
    ----------
    data :
        Train or test or dev dataset (after extracting fetures)
    PredictLabel : list
        list of all Prediction labels
    
    Returns
    -------
        Classification report and Confusion matrix
    """

    GoldLabel = extract_golds_from_data(data) # First extract Gold labels from data 

    label_set = sorted(set(GoldLabel)) # find uniqe lables in gold
    print(label_set)

    print('precision_recall_f1-score')
    report = calculate_precision_recall_f1score(GoldLabel, PredictLabel, label_set) # calculate_precision_recall_f1score

    print('Confusion matrix')
    cf_matrix = provide_confusion_matrix(GoldLabel, PredictLabel, label_set) # provide_confusion_matrix

    return report, cf_matrix # return report and cf_matrix
