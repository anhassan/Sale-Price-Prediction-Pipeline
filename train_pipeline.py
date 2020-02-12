import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import modules as mod
import pipeline as pp
import math
from sklearn.metrics import mean_squared_error
import config as conf




def save_pipeline(pipeline):
    
    # Naming the pipeline
    
    pipeline_model = '/reg_pipe.pkl'
    PIPELINE_PATH = conf.TRAINED_MODEL_DIR + pipeline_model
    
    # Dumping the pipeline to a directory
    
    joblib.dump(pipeline,PIPELINE_PATH)
    
    print ('Pipeline Trained and Saved....')
    


def run_training():
    
    
    # Training the model pipeline
    reg_pipe = pp.pipe.fit(conf.X_train,conf.y_train)
    
    # Making the pipeline presistent
    save_pipeline(reg_pipe)
    

    
    
if __name__ == '__main__':
    run_training()






