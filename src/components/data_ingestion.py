import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
sys.path.insert(0,'D:\Red_Wine_Quality_Prediction\src')
from logger import *
from exception import *
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
     raw_data=os.path.join('artifacts','raw.csv')
     train_data=os.path.join('artifacts','train.csv')
     test_data=os.path.join('artifacts','test.csv')

class DataIngestion:
     def __init__(self):
          self.data_ingestion_config=DataIngestionConfig()
     def initiate_data_ingestion(self):
            try:
                df=pd.read_csv('experiments\cleaned_data.csv')
                os.makedirs(os.path.dirname(self.data_ingestion_config.train_data),exist_ok=True)
                train_dataset,test_dataset=train_test_split(df,test_size=0.20,random_state=42)
                train_dataset.to_csv(self.data_ingestion_config.train_data)
                test_dataset.to_csv(self.data_ingestion_config.test_data)
                df.to_csv(self.data_ingestion_config.raw_data)
            except Exception as e:
                logging.info("ERROR OCCURED IN DATA INGESTION")
                raise CustomException(e,sys)
                 
if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()     

'''So, the code inside the if __name__ == '__main__': block will only execute if the script is run directly.
 If the script is imported as a module, this block will not execute.'''  

