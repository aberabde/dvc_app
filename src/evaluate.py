import os
import argparse
import pandas as pd
from get_data import read_params
from joblib import dump, load
import numpy as np
import sklearn.metrics as metrics
from utils.data_clean import get_df, save_matrix, save_json



def evaluate(config_path):
    config = read_params(config_path)
    model_path = config["train"]["model"]
    featurized_test_path = config['featurize']["featurized_test"]
    scores_path = config["evaluate"]["metrics"]


    model = load(model_path)
    matrix = load(featurized_test_path)

    labels = np.squeeze(matrix[:, 1].toarray())
    X = matrix[:, 2:]
    print(X)
    predictions_probabilities = model.predict_proba(X)
    pred = predictions_probabilities[:, 1]

    avg_prec = metrics.average_precision_score(labels, pred)
    roc_auc = metrics.roc_auc_score(labels, pred)
    
    #accuracy = metrics.accuracy_score(labels, pred)

    scores = {
    "avg_prec": avg_prec,
    "roc_auc" : roc_auc
    }
    print(scores)
    
    save_json(scores_path  , scores)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    evaluate(config_path=parsed_args.config)

