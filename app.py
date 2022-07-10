import os
import argparse
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.externals import joblib
from get_data import read_params
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from utils.data_clean import processed_data




# load the model from disk
# filename = 'nlp_model.pkl'
# clf = pickle.load(open(filename, 'rb'))
config = read_params(config_path)
model_path = config["train"]["model"]






app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    	if request.method == 'POST':
		message1 = request.form['message']
		message = processed_data(message1)
		data = [message]
		
		
		
		vect = cv.transform(data).toarray()
		
		
		
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    predict(config_path=parsed_args.config)

if __name__ == '__main__':
	app.run(debug=True)