# split the raw data 
# save it in data/processed folder
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params
import joblib
import numpy as np
import scipy.sparse as sparse




def split_and_saved_data(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test"] 
    train_data_path = config["split_data"]["train"]
    clean_data_path = config["preprocess_data"]["clean_data"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    df = pd.read_csv(clean_data_path, sep=";")
    train, test = train_test_split(
        df, 
        test_size=split_ratio,
        stratify=df.label, 
        random_state=random_state
        )

    print(df.head())
    print(df.head().index.tolist())
    
    print(df['text'].isnull().values.any())
    
    print(df[df['text'].isna()])   
    print(test.head())
    print(test.head().index.to_list())

    #train.set_index('Unnamed: 0',inplace=True)
    #test.set_index('Unnamed: 0',inplace=True)

    train.to_csv(train_data_path, sep=";", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=";",index=False,encoding="utf-8")

    #joblib.dump(train, train_data_path, compress=4)
    #joblib.dump(test, test_data_path, compress=4)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)
