import pandas as pd
import os
import sys
from logger import *
from exception import *
import pickle



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
