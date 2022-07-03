import os
import argparse
import pandas as pd
from get_data import read_params
from src.utils import *


def processed_and_saved_data(config_path):
    config = read_params(config_path)
    clean_data_path=config["prepare_data"]["clean_data"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]

    source = pd.read_csv(raw_data_path, sep=";")

    print("STARTING PREPROCESSING THE RAW TEXT ...\n")
    
    
    result = keep_alphanumeric(source)
    result = convert_lower_case(result)
    result = convert_numbers_to_text(result)
    result = remove_single_characters(result)
    result = lemmatize(result)
    result = remove_stop_words(result, stopwords.words('english'))
    
    
    print("FINISHED PREPROCESSING THE RAW TEXT ...")

    print( result.head())

    clean_news.to_csv(train_data_path, sep=";", encoding="utf-8")

    #joblib.dump(result, train_data_path)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    processed_and_saved_data(config_path=parsed_args.config)