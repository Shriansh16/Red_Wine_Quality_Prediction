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
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            X_train=train_data.iloc[:,:-1]
            y_train=train_data.iloc[:,-1]
            X_test=test_data.iloc[:,:-1]
            y_test=test_data.iloc[:,-1]
            pre=self.get_data_transformation()
            X_train_trans=pre.fit_transform(X_train)
            X_test_trans=pre.transform(X_test)
            X_train_trans=X_train_trans.toarray()
            X_test_trans=X_test_trans.toarray()
            final_train=np.c_[X_train_trans,np.array(y_train)]
            final_test=np.c_[X_test_trans,np.array(y_test)]
            save_object(pre,self.data_transforamtion_config.preprocessor_path)
            return (final_train,final_test)
        except Exception as e:
            logging.info("ERROR OCCURED IN DATA TRANSFORMATION STEP 2")
            raise CustomException(e,sys)
        


