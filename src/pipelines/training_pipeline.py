import os
import sys
sys.path.insert(0,'D:\Red_Wine_Quality_Prediction\src')
from components.data_ingestion import *
from components.data_transformation import *
from components.model_trainer import *
from utils import *

if __name__=='__main__':
    obj1=DataIngestion()
    train_path,test_path=obj1.initiate_data_ingestion()
    obj2=DataTransformation()
    train_arr,test_arr=obj2.initiate_data_transforamtion(train_path,test_path)
    obj3=Model_Trainer()
    obj3.initiate_training(train_arr,test_arr)