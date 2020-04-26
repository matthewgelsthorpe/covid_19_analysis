from model_loader.model_loader import CasesModel, DeathModel

c_model = CasesModel(model_location="model_data/models/rf_cases_trainsplit.pkl", model_json="model_data/json/cases_model.json", country_lookup="model_data/country_data_full.csv")
d_model = DeathModel(model_location="model_data/models/death_model.pkl", model_json="model_data/json/deathmodel_cols.json", country_lookup="model_data/death_model_full.csv")