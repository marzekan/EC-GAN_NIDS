from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
import numpy as np
import tensorflow as tf

# Disable scientific notation in numpy.
np.set_printoptions(precision=2, suppress=True)

labels = {
    0: 'BENIGN',
    1: 'Bot',
    2: 'DDoS',
    3: 'DoS_GoldenEye',
    4: 'DoS_Hulk',
    5: 'DoS_Slowhttptest',
    6: 'DoS_slowloris',
    7: 'FTPPatator',
    8: 'Heartbleed',
    9: 'Infiltration',
    10: 'PortScan',
    11: 'SSHPatator',
    12: 'Web_Attack_Brute_Force',
    13: 'Web_Attack_Sql_Injection',
    14: 'Web_Attack_XSS'
}

def make_prediction(model: tf.keras.models.Sequential,
                    test_data: pd.DataFrame,
                    test_lbl: pd.DataFrame):
    """Make single prediction from single model."""

    prediction = model.predict(test_data)

    prediction = [

        (labels[int(test_lbl[0][i])], labels[np.where(
            max_ == np.amax(max_))[0][0]], np.amax(max_))

        for i, max_ in zip(range(len(test_lbl)), prediction)
    ]

    prediction = pd.DataFrame(prediction, columns=[
        "Real", "Predicted", "Confidence"])

    return prediction


class PredMetrics:
    """Contains metrics for evaluating classification capabilities of a model."""

    def __init__(self):
        self.confusion_matrix = None
        self.classification_report = None
        self.fp_rates = None
        self.weighted_fpr = None
        self.detection_rates = None
        self.weighted_detection_rate = None
        self.fn_rates = None
        self.weighted_fnr = None
        self.false_alarm_rate = None

        
# In sklearn.metrics.multilabel_confusion_matrix the elements are as follows:
#     True Positives = [1][1]
#     True Negatives = [0][0]
#     False Positives = [0][1]
#     False Negatives = [1][0]
def get_prediction_metrics(prediction: pd.DataFrame):

    pm = PredMetrics()

    pm.confusion_matrix = multilabel_confusion_matrix(prediction.Real,
                                                      prediction.Predicted,
                                                      labels=prediction.Real.unique())

    pm.classification_report = pd.DataFrame(
        classification_report(
            prediction.Real,
            prediction.Predicted,
            output_dict=True,
            zero_division=0
        )).T

    # Individual FP rates for each class. FP / (FP+TN)
    pm.fp_rates = [cm[0][1] / (cm[0][1]+cm[0][0])
                   if (cm[0][1]+cm[0][0]) != 0 else 0
                   for cm in pm.confusion_matrix]

    class_sizes = prediction.Real.value_counts(sort=False)

    # Weighted False Positives Rate of a model.
    pm.weighted_fpr = sum([(cm[0][1] / (cm[0][1]+cm[0][0])) * class_size if (cm[0][1]+cm[0][0]) != 0 else 0
                           for cm, class_size in zip(pm.confusion_matrix, class_sizes)]) / sum(class_sizes)
    
    # Sensitivity, formula: TP / (TP+FN) 
    pm.detection_rates = [cm[1][1] / (cm[1][1]+cm[1][0])
                          if (cm[1][1]+cm[1][0]) != 0 else 0
                          for cm in pm.confusion_matrix]
    
    pm.weighted_detection_rate = sum([(cm[1][1] / (cm[1][1]+cm[1][0])) * class_size if (cm[1][1]+cm[1][0]) != 0 else 0 
                                      for cm, class_size in zip(pm.confusion_matrix, class_sizes)]) / sum(class_sizes)
    
    # Formula: FN / (FN+TP)
    pm.fn_rates = [cm[1][0] / (cm[1][0]+cm[1][1])
                   if (cm[1][0]+cm[1][1]) != 0 else 0
                   for cm in pm.confusion_matrix]
    
    pm.weighted_fnr = sum([(cm[1][0] / (cm[1][0]+cm[1][1])) * class_size if (cm[1][0]+cm[1][1]) != 0 else 0
                           for cm, class_size in zip(pm.confusion_matrix, class_sizes)]) / sum(class_sizes)
    
    pm.false_alarm_rate = (pm.weighted_fpr + pm.weighted_fnr) / 2

    return pm