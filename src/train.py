import os
import argparse
import pandas as pd
from get_data import read_params
from joblib import dump, load
import numpy as np
from utils.data_clean import get_df, save_matrix
from sklearn.ensemble import RandomForestClassifier



def train(config_path):
    config = read_params(config_path)
    model_path = config["train"]["model"]
    random_state = config["base"]["random_state"]
    n_est =config["train"]["n_est"]
    min_split = config["train"]["min_split"]
    n_jobs =config["train"]["n_jobs"]

    featurized_train_path = config['featurize']["featurized_train"]


    matrix = load(featurized_train_path)

    labels = np.squeeze(matrix[:, 1].toarray())
    X = matrix[:, 2:]

    #print(labels)
    #print(X)

    model = RandomForestClassifier(
        n_estimators=n_est,
        min_samples_split=min_split,
        n_jobs=n_jobs,
        random_state=random_state
    )
    model.fit(X, labels)

    dump(model, model_path)




if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train(config_path=parsed_args.config)