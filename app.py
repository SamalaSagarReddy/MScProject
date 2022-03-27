#import the library files
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

#initialise the flask
app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/result.html/')
def result():
	return render_template("result.html")  


@app.route('/prediction.html/', methods = ['POST','GET'])
def prediction():
	return render_template("prediction.html")

# @app.route('/predict',methods=['POST', 'GET'])
# def predict():
#     if request.method=='POST':
#         result=request.form

# 		#Prepare the feature vector for prediction
#         pkl_file = open('cat', 'rb')
#         index_dict = pickle.load(pkl_file)
#         new_vector = np.zeros(len(index_dict))

#         try:
#         	new_vector[index_dict['text'+str(result['text'])]] = 1
#         except:
#             pass

#         new_vector = new_vector.reshape(-1,1)
#         pkl_file = open('model_0.pkl', 'rb')
#         NBmodel = pickle.load(pkl_file)
#         prediction = NBmodel.predict(new_vector)
        
#         return render_template('result.html',prediction_text=prediction)


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        result=request.form
        model = pickle.load(open('model_0.pkl','rb'))
        features = [x for x in request.form.values()]
        #final_features = [np.array(features)]
        prediction = model.predict(features)
        output = prediction[0]
        return render_template("prediction.html" ,prediction_text=output)


if __name__ == '__main__':
	app.run(debug=True) 