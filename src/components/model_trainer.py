import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0,'D:\Red_Wine_Quality_Prediction\src')
from logger import *
from exception import *
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass
class Model_trainer_config:
    model_path=os.path.join('artifacts','model.pkl')

class Model_Trainer:
    def __init__(self):
        self.model_trainer_config=Model_trainer_config
    def initiate_training(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            models={
                'Logistic_regression':LogisticRegression(),
                'knn_c':KNeighborsClassifier(),
                'random_forest_classifier':RandomForestClassifier(),
                'svc':SVC(),'gradient_boosting_classifier':GradientBoostingClassifier(),
                'ada_boost_classifier':AdaBoostClassifier(),'decision_tree_classifier':DecisionTreeClassifier()
            }