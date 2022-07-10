import os
import argparse
from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle
# from sklearn.externals import joblib
from src.get_data import read_params
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from src.utils.data_clean import processed_data




	

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])



def predict():

	if request.method == 'POST':

		message = request.form['message']

		message= pd.Series(message)
# message = message.to_frame()[0] 
		data = processed_data(message)
		data_words = np.array(data.str.lower().values.astype("U"))
		# print(data_words)
		# data = [message]
		
		bag_of_words = load(os.path.join("saved_models", "bag_of_words.pkl"))
		tfidf = load(os.path.join("saved_models", "tfidf.pkl"))
# bag_of_words = CountVectorizer()		
		vect = bag_of_words.transform(data_words)
		vect_tfidf = tfidf.transform(vect)
		
		model = load(os.path.join("saved_models", "model.pkl"))
		my_prediction = model.predict(vect_tfidf)
# print(my_prediction[0])
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
	