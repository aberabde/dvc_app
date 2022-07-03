import os
from get_data import read_params, get_data
import argparse



def load_and_save(config_path):
    config = read_params(config_path)
    news = get_data(config_path)
    print(f"100% des observations: {news.shape[0]}")
    sample = config["load_data"]["sample_frac"]
    random_state = config["base"]["random_state"]
    news = news.sample(frac = sample , random_state = random_state)

    news.reset_index(drop=True, inplace=True)

    print(f" 1% des observations: {news.shape[0]}")
    
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    
    news.to_csv(raw_data_path, sep=";")





if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)