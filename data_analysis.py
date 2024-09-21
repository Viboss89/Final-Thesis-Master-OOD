import csv 
import matplotlib.pyplot as plt
import sklearn.metrics
import pandas as pd
import numpy as np

path = "Results_ood.csv"
standard_model = True

# Script to read .csv of inference.py and extract data

def score_csv(path, standard_model):
    """
    path: path of the csv file
    standar_model: if True, it doesn't consider if the sample is reliable or unreliable
    """

    df = pd.read_csv(path)
    if not standard_model:
        counts = df["Reliability"].value_counts()
        y_true = df["Imagefiles"].where(df["Reliability"]=="Reliable", other=-1.0).to_numpy()
        y_pred = df["Prediction results"].where(df["Reliability"]=="Reliable", other=-1.0).to_numpy()
        y_true = np.array([int(float(y_true[i])) for i in range(len(y_true))])
        y_pred = np.array([int(float(y_pred[i])) for i in range(len(y_pred))])
        y_true = y_true[y_true >= 0]
        y_pred = y_pred[y_pred >= 0]

        print("Reliable inferences:", counts)

    else:
        y_true = df["Imagefiles"].to_numpy()
        y_pred = df["Prediction results"].to_numpy()


    report = sklearn.metrics.accuracy_score(y_true, y_pred)
    cl = sklearn.metrics.classification_report(y_true, y_pred)
    print(report, cl)

score_csv("Results_ood_OCTID.csv", False)
#score_csv("Results_standard_model_kellman.csv", True)