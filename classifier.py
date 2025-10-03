import json
import os
import pickle
import re
import time
from dataclasses import dataclass, field


from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import pandas as pd
from feat_generator import FeatGenerator

@dataclass
class Clasifier:
    gender_model_path: str = None
    age_model_path: str = None
    bot_user_model_path: str = None
    scaler_path: str = None
    mode: str = 'Train'
    users: list = None

    def __post_init__(self):
        if self.mode == 'Predict':
            self._load_models()
            self.is_fit = True
        else:
            self.is_fit = False

    def _load_models(self):
        self.gender_model = joblib.load(self.gender_model_path)
        self.real_user_model = joblib.load(self.bot_user_model_path)
        self.scaler = joblib.load(self.scaler_path)

        with open(self.age_model_path, 'rb') as handle:
            self.age_models = pickle.load(handle)

        self.models = {'Male Pct': self.gender_model, 'Bot User Pct': self.real_user_model, 'Age Pcts': self.age_models}

    def generate_model_data(self, users, table_paths, fill_na):
        feat_gen = FeatGenerator(
            users=users,
            fill_na=fill_na,
            table_save_paths=table_paths,
            mode=self.mode
            )
        
        if self.mode == 'Train':
            feat_gen.get_target_data()
            feat_gen.get_all_feats()
            feat_gen.combine_feats()
            
            df_gender = feat_gen.generate_gender_train_data()
            df_age = feat_gen.generate_age_train_data()
            df_bot = feat_gen.generate_bot_train_data()
            self.data = {
                'feats': feat_gen.df_feats_all.copy(),
                'gender': df_gender,
                'age': df_age,
                'bot': df_bot
            }
        
        else:
            feat_gen.get_all_feats()
            feat_gen.combine_feats()
            self.data = {
                'feats': feat_gen.df_feats_all.copy()
            }
    
    def predict(self, data):
        pred_dict = {key: {'Male Pct': np.nan, 'Bot User Pct': np.nan, 'age1': np.nan, 'age2': np.nan, 'age3': np.nan, 'age4': np.nan, 'age5': np.nan} for key in self.users}
        X = self.scaler.transform(data.values)
        pred_dict_2 = {}
        for model in self.models:
            if model == 'Age Pcts':
                for age in self.models[model]:
                    pred_dict_2[age] = self.models[model][age].predict(X)
            else:
                pred_dict_2[model] = self.models[model].predict(X)
        pred_dict.update(pd.DataFrame(pred_dict_2, index=data.index).T.to_dict())
        return pred_dict
    
    def train_scaler(self, data):
        self.scaler = MinMaxScaler()
        self.scaler = self.scaler.fit(data.values)
        return self.scaler
    
    def train_gender_model(self, data, K):
        X_train = data.drop('male', axis=1).values
        y_train = data.male.values
        X_train = self.scaler.transform(X_train)
        self.knn_gender = KNeighborsRegressor(n_neighbors=K)
        self.knn_gender.fit(X_train,y_train)
        return self.knn_gender

    def train_age_model(self, data, K):
        self.age_models = {}
        X_train = data.drop(['age1', 'age2', 'age3', 'age4', 'age5'], axis=1).values
        X_train = self.scaler.transform(X_train)
        y_trains = {a: data[a].values for a in ['age1', 'age2', 'age3', 'age4', 'age5']}

        for y_train in y_trains:
            knn = KNeighborsRegressor(n_neighbors=K)
            knn.fit(X_train, y_trains[y_train])
            self.age_models[y_train] = knn
        
        return self.age_models
        
    def train_bot_model(self, data, K):
        X_train = data.drop('bot_user', axis=1).values
        y_train = data.bot_user.values
        X_train = self.scaler.transform(X_train)
        self.knn_bot = KNeighborsRegressor(n_neighbors=K)
        self.knn_bot.fit(X_train,y_train)
        return self.knn_bot








        

        



