import os
import argparse
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from get_data import read_params
from utils.data_clean import *
import random


    
def saved_data(config_path):
    config = read_params(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"] # in csv
    clean_data_path = config["preprocess_data"]["clean_data"] # out csv
    
    news = pd.read_csv(raw_data_path, sep=";")
    print(news.iloc[10])
    
    news['text']   = processed_data(news['text'] )
    news['title']  = processed_data(news['title'] )


    

    #clean_news.to_csv(clean_data_path, sep=";", encoding="utf-8")
    #df = joblib.load('df.joblib')
    #joblib.dump(result, train_data_path)
    clean_news= news.copy()

    clean_news['label'] = clean_news['label'].replace(['true', 'fake'], [1, 0], inplace=False)
    None
    # True:1 , False:0
 
    clean_news.reset_index(drop=True, inplace=True)

    print(clean_news.head())
    print(clean_news.head().index.tolist())

    clean_news.to_csv(clean_data_path, sep=";", index=True, encoding="utf-8")


    random_number = 10

    print("Inspection visuelle de l'observation portant le numéro {index}:\n".format(index = random_number))
    print("Title: {title}\n".format(title = clean_news.title[random_number]))
    print("Contenu: {text}\n".format(text = clean_news.text[random_number]))
    print("Clasification: {label}".format(label = clean_news.label[random_number]))



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    saved_data(config_path=parsed_args.config)




