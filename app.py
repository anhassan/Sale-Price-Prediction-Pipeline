import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib
import config as conf

pipeline_model = '/reg_pipe.pkl'

MODEL_DIR = conf.TRAINED_MODEL_DIR + pipeline_model

loaded_pipeline  = joblib.load(MODEL_DIR)
print("model loaded")
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    y_pred = loaded_pipeline.predict(conf.X_test)
    output = round(y_pred[0],2)

    return render_template('index.html', prediction_text='Root mean square error : {}'.format(output))
    

if __name__ == "__main__":
    app.run(host='0.0.0.0')