import pandas as pd
import os
import sys
from logger import *
from exception import *
import pickle
from sklearn.metrics import accuracy_score



def save_object(obj,path):
    try:
        dir_path=os.path.dirname(path)
        os.makedirs(dir_path,exist_ok=True)
        with open(path,'wb') as p:
            pickle.dump(obj,p)
    except Exception as e:
        logging.info("ERROR OCCURED IN SAVING THE OBJECT")
        raise CustomException(e,sys)
def open_object(path):
    try:
        with open(path,'rb') as obj:
            return pickle.load(obj)
    except Exception as e:
        logging.info("ERROR OCCURED IN LOADING THE OBJECT")
        raise CustomException(e,sys)
    
def evaluate_models(models,X_train,X_test,y_train,y_test):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            predictions=model.predict(X_test)
            score=accuracy_score(y_test,predictions)
            report[list(models.keys())[i]]=score
        return report
    except Exception as e:
        logging.info("ERROR OCCURED DURING MODEL EVALUATION")
        raise CustomException(e,sys)

