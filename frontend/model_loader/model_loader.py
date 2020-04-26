import pandas as pd
import numpy as np
from sklearn.externals import joblib
import json
import os
from datetime import datetime
from copy import deepcopy


class ModelFunctions:
    
    def __init__(self, country_lookup, model_type):
        
        self.model_type = model_type
        self.country_lookup = country_lookup
        self.country_df = pd.read_csv(country_lookup)
        self.country_df["dateRep"] = pd.to_datetime(self.country_df["dateRep"])
        self.country_start_end = self.country_df[self.country_df.flag >0][["dateRep",
                                                                           self.model_type,
                                                                           "countriesAndTerritories",
                                                                           "flag"]].groupby("countriesAndTerritories",
                                                                                            as_index=False, ).apply(lambda x: self.first_last(x))
        self.country_start_end.reset_index(inplace= True, drop=True)
        
    @staticmethod       
    def first_last(df):
        first = df.iloc[0:1,:]
        last = df.iloc[-1:,:]
        df = pd.concat([first, last])
        return df
        
    @staticmethod
    def load_columns_names(json_location: str):
        with open(json_location, "r") as file:
            headers = json.load(file)["features"]
        return headers
            
    @staticmethod
    def load_model(model_location: str):
        model = joblib.load(model_location, "r")
        return model
    
    def get_flag(self, date: str, country: str):
        date = datetime.strptime(date, "%Y-%m-%d")
        df = self.country_df[(self.country_df.countriesAndTerritories == country)&
                             (self.country_df.dateRep == date)]
        if len(df) > 0:
            return {self.model_type: df[self.model_type].iloc[0]}
        else:
            max_date = self.country_df[self.country_df.countriesAndTerritories == country].dateRep.max()
            min_date = self.country_df[self.country_df.countriesAndTerritories == country].dateRep.min()
            
            max_event = self.country_df[self.country_df.countriesAndTerritories == country][self.model_type].max()
            if date > max_date:
                df = self.country_start_end[self.country_start_end.countriesAndTerritories ==country]
                if len(df) == 0:
                    return {self.model_type: max_event}
                else:
                    diff = date - df.dateRep.max()
                    cases_flag = df.flag.max()
                    cases_flag = diff.days + cases_flag
                    return{"flag": cases_flag}
            elif date < min_date:
                return {self.model_type: 0}
                
                
class CasesModel(ModelFunctions):
    
    def __init__(self, model_location: str, model_json: str, country_lookup: str):
        
        ModelFunctions.__init__(self, country_lookup, model_type="cases")
        self.model_location = model_location
        self.model_json = model_json
        self.model_headers = self.load_columns_names(model_json)
        self.model = self.load_model(model_location)
        self.model_array = np.zeros(len(self.model_headers))
        
    def gen_prediction(self, date, country: str):
        flag = self.get_flag(date, country)
        if "flag" in flag.keys():
            n_day = flag["flag"]
            print(f"flag value {n_day}")
            feature_array = deepcopy(self.model_array)
            country_col = f"countriesAndTerritories_{country}"

            index_country = self.model_headers.index(country_col)
            index_flag = self.model_headers.index("flag")
            feature_array[index_country] = 1
            feature_array[index_flag] = n_day
            self.feature_array = feature_array
            prediction = self.model.predict(feature_array.reshape(1, -1))
            return int(np.round(prediction))
        else:
            return flag["cases"]
        
        
class DeathModel(ModelFunctions):

    def __init__(self, model_location: str, model_json: str, country_lookup: str):
        
        ModelFunctions.__init__(self, country_lookup, model_type="deaths")
        self.model_location = model_location
        self.model_json = model_json
        self.model_headers = self.load_columns_names(model_json)
        self.model = self.load_model(model_location)
        self.model_array = np.zeros(len(self.model_headers))
        
    def gen_prediction(self, date, country: str):
        flag = self.get_flag(date, country)
        if "flag" in flag.keys():
            n_day = flag["flag"]
            feature_array = deepcopy(self.model_array)
            country_col = f"Countries and territories_{country}"

            index_country = self.model_headers.index(country_col)
            index_flag = self.model_headers.index("flag")
            feature_array[index_country] = 1
            feature_array[index_flag] = n_day
            prediction = self.model.predict(feature_array.reshape(1, -1))
            return int(np.round(prediction))
        else:
            return flag["deaths"]
        
        