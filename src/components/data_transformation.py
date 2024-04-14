import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0,'D:\Red_Wine_Quality_Prediction\src')
from utils import *
from logger import *
from exception import *
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transforamtion_config=DataTransformationConfig()

    def get_data_transformation(self):
        try:
            num_columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates', 'alcohol']
            preprocessor=ColumnTransformer(['num_transformer',StandardScaler(),num_columns])
            return preprocessor
        except Exception as e:
            logging.info("ERROR OCCURED IN PREPROCESSOR STEP 1")
            raise CustomException(e,sys)
    def initiate_data_transforamtion(self,train_path,test_path):
        try:
