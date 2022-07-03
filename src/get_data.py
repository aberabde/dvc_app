# read params
# process
# return raw data as news.csv or DF 
import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    # print(config)
    data_path_1 = config["data_source"]["s3_source_1"]
    data_path_2 = config["data_source"]["s3_source_2"]
    random_state = config["base"]["random_state"]

    real_news = pd.read_csv(data_path_2)
    real_news['label'] = 'true'
    fake_news = pd.read_csv(data_path_1)
    fake_news['label'] = 'fake'

    fake_news = fake_news.sample(n=real_news.shape[0],random_state=random_state)
    
    news = pd.concat([real_news, fake_news])
    
    #print(news.head(1))
    ## df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    return news



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)