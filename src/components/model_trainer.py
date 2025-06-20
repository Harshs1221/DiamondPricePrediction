import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from src.utils import evaluate_model


from dataclasses import dataclass
import sys, os


@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting the dependent and independent variable from train and test arr")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {

            'LinearRegression': LinearRegression(),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'ElasticNet': ElasticNet(),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        }
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n=================================================================================\n')
            logging.info(f"Model Report: {model_report}")

            # to get the best model score form the dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model name {best_model_name}, R2 score {best_model_score}")
            print('\n=================================================================================\n')
            logging.info(f"Best Model Found, Model name {best_model_name}, R2 score {best_model_score}")

            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

        except Exception as e:
            logging.info("Exception occured in the initiate data transformation")
            raise CustomException(e,sys)
