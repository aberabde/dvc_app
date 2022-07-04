import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params
from joblib import dump, load
import numpy as np
import scipy.sparse as sparse
from utils.data_clean import get_df, save_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer




def featurize(config_path):
    config = read_params(config_path)
    clean_train_path = config["split_data"]["train_path"]
    clean_test_path = config['split_data']["test_path"]
    featurized_train_path = config['featurize']["featurized_train"]
    featurized_test_path =  config['featurize']["featurized_test"]
    max_features = config["featurize"]["max_features"]
    n_grams = config["featurize"]["n_grams"]
    
    # train

    df_train = get_df(clean_train_path, sep=";")
    print(df_train.head())
    train_words = np.array(df_train.text.str.lower().values.astype("U"))
    # print(train_words)
    bag_of_words = CountVectorizer(
        #stop_words="english",
        max_features=max_features,
        ngram_range=(1, n_grams)
    )

    bag_of_words.fit(train_words)
    train_words_binary_matrix = bag_of_words.transform(train_words)

    tfidf = TfidfTransformer(smooth_idf=False)
    tfidf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)

    # call a function to save this matrix
    save_matrix(df=df_train, text_matrix=train_words_tfidf_matrix, out_path=featurized_train_path)


    # test

    df_test = get_df(clean_test_path, sep=";")
    test_words = np.array(df_test.text.str.lower().values.astype("U"))
    test_words_binary_matrix = bag_of_words.transform(test_words)
    test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)
    # call a function to save this matrix
    save_matrix(df=df_test, text_matrix=test_words_tfidf_matrix, out_path=featurized_test_path)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    featurize(config_path=parsed_args.config)