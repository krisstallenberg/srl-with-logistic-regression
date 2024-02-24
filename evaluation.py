# I used 'provide_confusion_matrix' , 'calculate_precision_recall_f1score' , 'evaluation_model' functions from previous term
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

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
def evaluation_model(GoldLabel, PredictLabel): # get gold and predict
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

    label_set = sorted(set(GoldLabel)) # find uniqe lables in gold
    print(label_set)

    print('precision_recall_f1-score')
    report = calculate_precision_recall_f1score(GoldLabel, PredictLabel, label_set) # calculate_precision_recall_f1score

    print('Confusion matrix')
    cf_matrix = provide_confusion_matrix(GoldLabel, PredictLabel, label_set) # provide_confusion_matrix

    return report, cf_matrix # return report and cf_matrix
