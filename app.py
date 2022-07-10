import os
import argparse
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
# from sklearn.externals import joblib
from src.get_data import read_params
# from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from src.utils.data_clean import processed_data




# load the model from disk
# filename = 'nlp_model.pkl'
# clf = pickle.load(open(filename, 'rb'))








app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict(config_path):
	
	model_path = config["train"]["model"]
	max_features = config["featurize"]["max_features"]
	n_grams = config["featurize"]["n_grams"]



	if request.method == 'POST':
		message = request.form['message']
		message = processed_data(message)
		data = [message]
		bag_of_words = CountVectorizer(
    	# stop_words="english",
    		max_features=max_features,
    		ngram_range=(1, n_grams))
		vect = bag_of_words.transform(data).toarray()
		
		
		
		my_prediction = model_path.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    predict(config_path=parsed_args.config)

if __name__ == '__main__':
	app.run(debug=True)
	