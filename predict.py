import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor


def column_ratio(X):
    return X[:, [0]] / (X[:, [1]] + 1e-6)

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def get_cluster_feature_names(transformer, feature_names_in):
    n_clusters = len(transformer.kw_args["Y"])
    return [f"Cluster {i} similarity" for i in range(n_clusters)]


if __name__ == "__main__":
    
    print("LoadinmODEL :pppp")
    
    try:
        model = joblib.load("my_california_housing_model.pkl")
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: 'my_california_housing_model.pkl' not found.")
        print("Please run the 'train_model.py' script first to create the model file.")
        exit()
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        exit()

    
    new_data = pd.DataFrame({
        'longitude': [-122.23, -118.24, -121.89],
        'latitude': [37.88, 34.05, 37.34],
        'housing_median_age': [41.0, 25.0, 52.0],
        'total_rooms': [880.0, 5612.0, 1467.0],
        'total_bedrooms': [129.0, 1283.0, 190.0],
        'population': [322.0, 3174.0, 496.0],
        'households': [126.0, 1192.0, 177.0],
        'median_income': [8.3252, 3.5349, 7.2574],
        'ocean_proximity': ['NEAR BAY', '<1H OCEAN', 'NEAR BAY']
    })
    


    print(f"\nMaking predictions for {len(new_data)} new districts...")

 
    predictions = model.predict(new_data)
    
    print("Predictions complete!")


    predictions_df = pd.DataFrame(predictions, columns=["Predicted Price"])
    
    
    results_df = pd.concat([new_data.reset_index(drop=True), predictions_df], axis=1)

    print("\n--- Prediction Results ---")
    
    print(results_df.to_string())

